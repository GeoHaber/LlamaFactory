#!/usr/bin/env python
# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Zena — Dean of the University of Distillation.

Zena is a Gemma 4 GGUF chatbot that:
  1. Helps users pick teacher models by scanning for local GGUFs
  2. Credentials-checks each teacher (name heuristic + probe)
  3. Runs cross-examination: teachers evaluate each other's answers
  4. Builds a peer-respect matrix showing who trusts whom
  5. Acts as the hallucination judge for borderline SILVER samples

Backend: zen_core_libs InProcessAdapter (direct GGUF, LRU-cached).
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections.abc import Iterator
from pathlib import Path

# ── zen_core_libs — single source of truth for shared infrastructure ──────────
from zen_core_libs.eval import extract_score  # xray: ignore[SEC-015]
from zen_core_libs.llm import get_inprocess_adapter  # xray: ignore[SEC-015]


SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parent


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_GGUF_DIRS = [
    "C:/AI/Models",
    os.path.expanduser("~/AI/Models"),
    os.path.expanduser("~/.cache/llama.cpp"),
]

ZENA_SYSTEM_PROMPT = """\
You are Zena, the Dean of the University of Distillation.
You help users select teacher models for knowledge distillation.
You speak warmly but precisely. You are an expert in LLM capabilities.
When evaluating teacher answers, be rigorous and fair.
Keep responses concise — under 120 words unless asked for detail."""

# Cross-examination prompt template
CROSS_EXAM_PROMPT = """\
You are a peer reviewer at an academic conference.
A colleague was asked: "{prompt}"
Their answer was:
---
{answer}
---
Rate this answer on three dimensions (1-10 each):
- accuracy: Is it factually correct?
- completeness: Does it fully address the question?
- clarity: Is it well-structured and easy to understand?

Respond ONLY with JSON: {{"accuracy": N, "completeness": N, "clarity": N}}"""

# Hallucination judge prompt
HALLUCINATION_JUDGE_PROMPT = """\
You are a hallucination detector. Given a question and an answer, identify any hallucinated content.
Hallucination = claims that are fabricated, not supported by the question, or factually wrong.

Question: "{prompt}"
Answer: "{answer}"

Respond with JSON:
{{"hallucination_score": N, "hallucinated_claims": ["claim1", ...], "verdict": "clean|minor|major"}}

hallucination_score: 0 (no hallucination) to 10 (entirely fabricated).
Only list actual hallucinated claims, not opinions or subjective statements."""

# Capability description for Zena's chat
CAPABILITY_DESCRIPTIONS = {
    "chat": "general conversation and Q&A",
    "reasoning": "logical deduction, puzzles, step-by-step analysis",
    "coding": "programming, debugging, code review",
    "math": "calculus, algebra, statistics, proofs",
    "translation": "multilingual translation",
    "ocr": "image/text extraction (vision models)",
    "safety": "content moderation and safety guardrails",
    "creative": "poetry, storytelling, creative writing",
}


# ---------------------------------------------------------------------------
# ZenaDean — the chatbot engine
# ---------------------------------------------------------------------------

class ZenaDean:
    """The Dean of the University of Distillation.

    Wraps a Gemma 4 GGUF model as a persistent chatbot that manages
    teacher enrollment, credentialing, cross-examination, and judging.
    """

    def __init__(
        self,
        zena_gguf: str,
        max_models: int = 8,
        n_ctx: int = 8192,
    ):
        self.zena_gguf = zena_gguf
        self.adapter = get_inprocess_adapter(max_models=max_models, default_n_ctx=n_ctx)
        self.history: list[dict[str, str]] = []

        # Enrolled teachers: {name: {gguf, capabilities, profile, peer_scores}}
        self.teachers: dict[str, dict] = {}
        # Peer respect matrix: {judge_name: {subject_name: avg_score}}
        self.peer_matrix: dict[str, dict[str, float]] = {}

    # ── Chat interface ──────────────────────────────────────────

    def chat(self, user_msg: str, max_tokens: int = 300) -> str:
        """Send a message to Zena and get a response."""
        self.history.append({"role": "user", "content": user_msg})
        reply = self.adapter.chat(  # xray: ignore[LLM-003]
            self.zena_gguf,
            system=ZENA_SYSTEM_PROMPT,
            messages=self.history[-20:],  # keep last 20 turns
            max_tokens=max_tokens,
            temperature=0.4,
        )
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def chat_stream(self, user_msg: str, max_tokens: int = 300) -> Iterator[str]:
        """Stream a response from Zena."""
        self.history.append({"role": "user", "content": user_msg})
        full = ""
        for chunk in self.adapter.chat_stream(
            self.zena_gguf,
            system=ZENA_SYSTEM_PROMPT,
            messages=self.history[-20:],
            max_tokens=max_tokens,
            temperature=0.4,
        ):
            full += chunk
            yield chunk
        self.history.append({"role": "assistant", "content": full})

    def reset_history(self):
        self.history.clear()

    # ── GGUF discovery ──────────────────────────────────────────

    @staticmethod
    def discover_gguf_models(extra_dirs: list[str] | None = None) -> list[dict]:
        """Scan common directories for .gguf files.

        Returns list of {path, name, size_gb}.
        """
        dirs = list(DEFAULT_GGUF_DIRS)
        if extra_dirs:
            dirs.extend(extra_dirs)

        found = []
        seen = set()
        for d in dirs:
            pattern = os.path.join(d, "**", "*.gguf")
            for path in glob.glob(pattern, recursive=True):
                path = os.path.normpath(path)
                if path in seen:
                    continue
                seen.add(path)
                try:
                    size_gb = os.path.getsize(path) / (1024**3)
                except OSError:  # xray: ignore[QUAL-002]
                    size_gb = 0
                found.append({
                    "path": path,
                    "name": Path(path).stem,
                    "size_gb": round(size_gb, 1),
                })
        found.sort(key=lambda x: x["name"].lower())
        return found

    # ── Teacher enrollment ──────────────────────────────────────

    def enroll_teacher(self, gguf_path: str, capabilities: dict[str, float] | None = None) -> dict:
        """Add a teacher to the university.

        Args:
            gguf_path: path to the .gguf model file
            capabilities: optional pre-computed capability scores

        Returns:
            Teacher info dict.
        """
        name = Path(gguf_path).stem

        # Name-based capability inference (instant)
        if capabilities is None:
            sys.path.insert(0, str(SCRIPTS_DIR))
            from teacher_profiler import infer_capabilities_by_name
            capabilities = infer_capabilities_by_name(gguf_path)

        # Determine top capabilities
        top_caps = sorted(
            [c for c, s in capabilities.items() if s >= 0.3],
            key=lambda c: capabilities[c],
            reverse=True,
        )

        teacher = {
            "gguf": gguf_path,
            "capabilities": capabilities,
            "top_capabilities": top_caps or ["chat"],
            "peer_scores": {},  # filled by cross-exam
            "enrolled": True,
        }
        self.teachers[name] = teacher
        return teacher

    def remove_teacher(self, name: str) -> bool:
        """Remove a teacher from the university."""
        if name in self.teachers:
            del self.teachers[name]
            self.peer_matrix.pop(name, None)
            for judge_scores in self.peer_matrix.values():
                judge_scores.pop(name, None)
            return True
        return False

    def get_enrollment_summary(self) -> str:
        """Markdown summary of enrolled teachers."""
        if not self.teachers:
            return "No professors enrolled yet."

        lines = [f"**{len(self.teachers)} professor(s) enrolled:**\n"]
        for name, info in self.teachers.items():
            caps = ", ".join(info["top_capabilities"][:3]) or "general"
            lines.append(f"- **{name}** — {caps}")

        # Capability coverage
        all_caps: dict[str, int] = {}
        for info in self.teachers.values():
            for c in info["top_capabilities"]:
                all_caps[c] = all_caps.get(c, 0) + 1
        if all_caps:
            coverage = ", ".join(f"{c} ({n})" for c, n in sorted(all_caps.items(), key=lambda x: -x[1]))
            lines.append(f"\n**Coverage:** {coverage}")

        return "\n".join(lines)

    # ── Cross-examination ───────────────────────────────────────

    def cross_examine(
        self,
        test_prompts: list[str] | None = None,
        max_tokens_answer: int = 200,
        max_tokens_judge: int = 150,
    ) -> dict[str, dict[str, float]]:
        """Run cross-examination: each teacher judges every other teacher's answers.

        1. Each teacher answers a small set of test prompts
        2. Every other teacher rates those answers (structured JSON scoring)
        3. Results build the peer-respect matrix

        Returns: {judge_name: {subject_name: avg_score (0-10)}}
        """
        if len(self.teachers) < 2:
            return {}

        if test_prompts is None:
            test_prompts = [
                "Explain what a neural network is in simple terms.",
                "Write a Python function to find the longest common subsequence of two strings.",
                "What causes the tides on Earth?",
            ]

        teacher_names = list(self.teachers.keys())

        # Step 1: Each teacher answers the test prompts
        answers: dict[str, list[str]] = {}  # {teacher_name: [answer1, answer2, ...]}
        for name in teacher_names:
            gguf = self.teachers[name]["gguf"]
            teacher_answers = []
            for prompt in test_prompts:
                try:
                    ans = self.adapter.chat(  # xray: ignore[LLM-003]
                        gguf,
                        system="You are a knowledgeable professor. Answer clearly and accurately.",
                        messages=[{"role": "user", "content": prompt}],  # xray: ignore[LLM-003]
                        max_tokens=max_tokens_answer,
                        temperature=0.3,
                    )
                    teacher_answers.append(ans)
                except Exception as exc:  # xray: ignore[QUAL-011]
                    teacher_answers.append(f"[Error: {exc}]")
            answers[name] = teacher_answers

        # Step 2: Each teacher judges every other teacher's answers
        peer_matrix: dict[str, dict[str, float]] = {}

        for judge_name in teacher_names:
            judge_gguf = self.teachers[judge_name]["gguf"]
            peer_matrix[judge_name] = {}

            for subject_name in teacher_names:
                if judge_name == subject_name:
                    continue

                scores_for_subject: list[float] = []
                for i, prompt in enumerate(test_prompts):
                    answer = answers[subject_name][i]
                    exam_prompt = CROSS_EXAM_PROMPT.format(
                        prompt=prompt, answer=answer,
                    )
                    try:
                        raw = self.adapter.chat(  # xray: ignore[LLM-003]
                            judge_gguf,
                            system="You are a rigorous academic peer reviewer. Output only JSON.",
                            messages=[{"role": "user", "content": exam_prompt}],  # xray: ignore[LLM-003]
                            max_tokens=max_tokens_judge,
                            temperature=0.1,
                        )
                        score = _parse_cross_exam_score(raw)
                        scores_for_subject.append(score)
                    except Exception:  # xray: ignore[QUAL-011]
                        scores_for_subject.append(5.0)  # neutral on error

                avg = sum(scores_for_subject) / len(scores_for_subject) if scores_for_subject else 5.0
                peer_matrix[judge_name][subject_name] = round(avg, 1)

        self.peer_matrix = peer_matrix

        # Store peer scores in teacher records
        for subject_name in teacher_names:
            received = [
                peer_matrix[j][subject_name]
                for j in teacher_names
                if j != subject_name and subject_name in peer_matrix.get(j, {})
            ]
            if received:
                self.teachers[subject_name]["peer_scores"] = {
                    "avg_received": round(sum(received) / len(received), 1),
                    "from": {
                        j: peer_matrix[j][subject_name]
                        for j in teacher_names
                        if j != subject_name and subject_name in peer_matrix.get(j, {})
                    },
                }

        return peer_matrix

    def get_peer_matrix_markdown(self) -> str:
        """Render the peer respect matrix as a markdown table."""
        if not self.peer_matrix:
            return "No cross-examination data yet."

        names = sorted(self.peer_matrix.keys())
        hdr = "| Judge \\ Subject | " + " | ".join(f"**{n}**" for n in names) + " |"
        sep = "|---| " + " | ".join("---" for _ in names) + " |"
        rows = []
        for judge in names:
            cells = []
            for subject in names:
                if judge == subject:
                    cells.append("—")
                else:
                    score = self.peer_matrix.get(judge, {}).get(subject, 0)
                    if score >= 7:
                        cells.append(f"**{score}** ✓")
                    elif score >= 5:
                        cells.append(f"{score}")
                    else:
                        cells.append(f"~~{score}~~")
            rows.append(f"| **{judge}** | " + " | ".join(cells) + " |")

        md = "### Peer Respect Matrix\n\n"
        md += "*Each professor rates others' answers (1-10). Higher = more respect.*\n\n"
        md += "\n".join([hdr, sep] + rows)

        # Summary: who is most respected?
        avg_received: dict[str, float] = {}
        for subject in names:
            received = [
                self.peer_matrix[j][subject]
                for j in names
                if j != subject and subject in self.peer_matrix.get(j, {})
            ]
            avg_received[subject] = round(sum(received) / len(received), 1) if received else 0
        best = sorted(avg_received.items(), key=lambda x: -x[1])
        md += f"\n\n**Most respected:** {best[0][0]} ({best[0][1]}/10)"
        if len(best) > 1:
            md += f" | **Least:** {best[-1][0]} ({best[-1][1]}/10)"

        return md

    # ── Hallucination judging ───────────────────────────────────

    def judge_hallucination(
        self,
        prompt: str,
        answer: str,
        max_tokens: int = 200,
    ) -> dict:
        """Use Zena (Gemma 4) to judge if an answer contains hallucinations.

        Returns: {hallucination_score: 0-10, hallucinated_claims: [...], verdict: str}
        """
        judge_prompt = HALLUCINATION_JUDGE_PROMPT.format(prompt=prompt, answer=answer)
        try:
            raw = self.adapter.chat(  # xray: ignore[LLM-003]
                self.zena_gguf,
                system="You are a factual accuracy checker. Output only valid JSON.",
                messages=[{"role": "user", "content": judge_prompt}],  # xray: ignore[LLM-003]
                max_tokens=max_tokens,
                temperature=0.1,
            )
            return _parse_hallucination_result(raw)
        except Exception as exc:  # xray: ignore[QUAL-011]
            return {"hallucination_score": 5, "hallucinated_claims": [], "verdict": "error", "error": str(exc)}

    # ── Zena's recommendations ──────────────────────────────────

    def recommend_teachers(self, goal: str) -> str:
        """Ask Zena to recommend teachers based on the user's goal.

        Uses Zena's LLM + the current enrollment + discovered models.
        """
        available = self.discover_gguf_models()
        enrolled_names = list(self.teachers.keys())

        context = f"Currently enrolled: {enrolled_names or 'none'}\n"
        context += "Available GGUFs on disk:\n"
        for m in available[:20]:  # cap at 20 to fit context
            context += f"  - {m['name']} ({m['size_gb']} GB)\n"
        context += f"\nUser's goal: {goal}\n"
        context += "Recommend which models to enroll as teachers. Explain why. "
        context += "Consider: coding teachers for code, reasoning teachers for logic, etc."

        return self.chat(context, max_tokens=400)

    # ── Serialization ───────────────────────────────────────────

    def save_state(self, path: str | Path) -> None:
        """Save the full university state to JSON."""
        state = {
            "zena_gguf": self.zena_gguf,
            "teachers": self.teachers,
            "peer_matrix": self.peer_matrix,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_state(cls, path: str | Path, **kwargs) -> ZenaDean:
        """Load university state from JSON."""
        with open(path, encoding="utf-8") as f:
            state = json.load(f)  # xray: ignore[PY-005]
        dean = cls(state["zena_gguf"], **kwargs)
        dean.teachers = state.get("teachers", {})
        dean.peer_matrix = state.get("peer_matrix", {})
        return dean


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_cross_exam_score(raw: str) -> float:
    """Extract average score from cross-exam JSON output.

    Expected: {"accuracy": N, "completeness": N, "clarity": N}
    Returns average of the three (0-10 scale).
    """
    # Try JSON parse first
    try:
        # Find JSON in the response
        match = re.search(r"\{[^{}]+\}", raw)
        if match:
            data = json.loads(match.group(0))
            vals = []
            for key in ("accuracy", "completeness", "clarity"):
                v = data.get(key, 5)
                if isinstance(v, (int, float)):
                    vals.append(max(0, min(10, float(v))))
            if vals:
                return round(sum(vals) / len(vals), 1)
    except (json.JSONDecodeError, ValueError):  # xray: ignore[QUAL-002]
        pass

    # Fallback: extract_score from zen_core_libs
    score = extract_score(raw)
    if score is not None:
        return round(score * 10, 1)  # normalize to 0-10

    # Last resort: look for any number
    numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", raw)
    if numbers:
        vals = [float(n) for n in numbers if 0 <= float(n) <= 10]
        if vals:
            return round(sum(vals) / len(vals), 1)

    return 5.0  # neutral default


def _parse_hallucination_result(raw: str) -> dict:
    """Parse hallucination judge output."""
    try:
        match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return {
                "hallucination_score": max(0, min(10, int(data.get("hallucination_score", 5)))),
                "hallucinated_claims": data.get("hallucinated_claims", []),
                "verdict": data.get("verdict", "unknown"),
            }
    except (json.JSONDecodeError, ValueError):  # xray: ignore[QUAL-002]
        pass

    # Fallback: check for keywords
    raw_lower = raw.lower()
    if "no hallucination" in raw_lower or "clean" in raw_lower:
        return {"hallucination_score": 0, "hallucinated_claims": [], "verdict": "clean"}
    if "major" in raw_lower or "fabricat" in raw_lower:
        return {"hallucination_score": 7, "hallucinated_claims": [], "verdict": "major"}

    return {"hallucination_score": 5, "hallucinated_claims": [], "verdict": "unknown"}


# ---------------------------------------------------------------------------
# CLI — for standalone testing
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Zena — Dean of the University of Distillation")
    parser.add_argument("--zena-gguf", required=True, help="Path to Gemma 4 GGUF for Zena.")
    parser.add_argument("--discover", action="store_true", help="Scan for available GGUF models.")
    parser.add_argument("--enroll", nargs="*", help="GGUF paths to enroll as teachers.")
    parser.add_argument("--cross-exam", action="store_true", help="Run cross-examination after enrollment.")
    parser.add_argument("--chat", action="store_true", help="Start interactive chat with Zena.")
    parser.add_argument("--save", default="", help="Save university state to this JSON path.")
    args = parser.parse_args()

    dean = ZenaDean(args.zena_gguf)

    if args.discover:
        models = dean.discover_gguf_models()
        print(f"\n=== Found {len(models)} GGUF models ===\n")  # xray: ignore[PY-004]
        for m in models:
            print(f"  {m['name']:40s}  {m['size_gb']:6.1f} GB  {m['path']}")  # xray: ignore[PY-004]
        print()  # xray: ignore[PY-004]

    if args.enroll:
        for path in args.enroll:
            teacher = dean.enroll_teacher(path)
            name = Path(path).stem
            caps = ", ".join(teacher["top_capabilities"][:3])
            print(f"  Enrolled: {name} ({caps})")  # xray: ignore[PY-004]
        print()  # xray: ignore[PY-004]
        print(dean.get_enrollment_summary())  # xray: ignore[PY-004]

    if args.cross_exam and len(dean.teachers) >= 2:
        print("\n=== Cross-Examination ===\n")  # xray: ignore[PY-004]
        dean.cross_examine()
        print(dean.get_peer_matrix_markdown())  # xray: ignore[PY-004]

    if args.chat:
        print("\n=== Chat with Zena (type 'quit' to exit) ===\n")  # xray: ignore[PY-004]
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if user_input.lower() in ("quit", "exit", "q"):
                break
            reply = dean.chat(user_input)
            print(f"Zena: {reply}\n")  # xray: ignore[PY-004]

    if args.save:
        dean.save_state(args.save)
        print(f"State saved to: {args.save}")  # xray: ignore[PY-004]

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
