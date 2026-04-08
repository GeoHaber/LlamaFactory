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

"""Teacher capability profiler — auto-detect what each teacher is good at.

Two detection methods:
  1. **Name heuristic** — fast, zero-cost, parses model filename for keywords
  2. **Mini-benchmark probe** — runs 5 tiny prompts per capability through
     each teacher and scores the responses (needs a running backend)

The probe outputs a capability matrix:
    teacher × capability → score (0-10)

This matrix feeds into prompt routing: each prompt is tagged with a capability,
and only teachers scoring ≥ threshold on that capability answer it.

Usage:
    # Name-only (instant)
    python scripts/teacher_profiler.py --teachers "C:/AI/Models/deepseek-r1.gguf" "C:/AI/Models/qwen2.5-coder.gguf"

    # With mini-benchmark probe (needs zen_core_libs)
    python scripts/teacher_profiler.py --teachers ... --probe --backend inprocess
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Capability taxonomy
# ---------------------------------------------------------------------------

CAPABILITIES = [
    "chat",
    "reasoning",
    "coding",
    "math",
    "translation",
    "ocr",
    "safety",
    "creative",
]

# ---------------------------------------------------------------------------
# Method 1: Name-based heuristic (instant, always available)
# ---------------------------------------------------------------------------

NAME_RULES: dict[str, list[str]] = {
    "chat": ["chat", "instruct", "-it-", "-it.", "assistant"],
    "reasoning": ["reason", "r1", "o1", "think", "deepthink", "qwq"],
    "coding": ["code", "coder", "starcoder", "codellama", "deepseek-coder", "codestral"],
    "math": ["math", "llemma", "wizardmath", "deepseekmath", "numina"],
    "translation": ["mt", "translate", "m2m", "nllb", "madlad", "tower"],
    "ocr": ["ocr", "vision", "vl", "llava", "qwen-vl", "kosmos", "minicpm-v", "paligemma"],
    "safety": ["guard", "safety", "moderation", "llamaguard", "shieldgemma"],
    "creative": ["creative", "story", "writer", "gemma-2", "gemma-4"],
}

# Well-known general-purpose model families — realistic baseline scores.
# Instruct/chat models from these families are capable of most tasks even
# when the filename doesn't include a capability keyword.
FAMILY_BASELINES: dict[str, dict[str, float]] = {
    # Gemma 4: strong chat, translation, creative, decent reasoning/math
    "gemma-4": {"chat": 1.0, "reasoning": 0.75, "translation": 0.80,
                "creative": 0.80, "math": 0.65, "safety": 0.60,
                "coding": 0.50, "ocr": 0.0},
    # Gemma 2: same profile, slightly lower reasoning
    "gemma-2": {"chat": 1.0, "reasoning": 0.65, "translation": 0.75,
                "creative": 0.75, "math": 0.55, "safety": 0.60,
                "coding": 0.45, "ocr": 0.0},
    # Qwen 2.5 Instruct: excellent all-rounder, strong coding + translation
    "qwen2.5": {"chat": 1.0, "reasoning": 0.80, "coding": 0.85,
                "translation": 0.85, "math": 0.80, "creative": 0.65,
                "safety": 0.55, "ocr": 0.0},
    # Mistral Instruct: solid chat, translation, moderate reasoning
    "mistral": {"chat": 1.0, "reasoning": 0.65, "translation": 0.75,
                "creative": 0.60, "math": 0.55, "coding": 0.65,
                "safety": 0.50, "ocr": 0.0},
    # DeepSeek: strong reasoning/coding
    "deepseek": {"chat": 0.90, "reasoning": 0.90, "coding": 0.85,
                 "math": 0.85, "translation": 0.55, "creative": 0.50,
                 "safety": 0.45, "ocr": 0.0},
    # LLaMA 3: strong all-rounder
    "llama-3": {"chat": 1.0, "reasoning": 0.75, "coding": 0.70,
                "translation": 0.65, "math": 0.65, "creative": 0.65,
                "safety": 0.55, "ocr": 0.0},
    # Phi-3/4: strong reasoning/math
    "phi": {"chat": 0.90, "reasoning": 0.80, "coding": 0.75,
            "math": 0.80, "translation": 0.55, "creative": 0.55,
            "safety": 0.55, "ocr": 0.0},
}


def infer_capabilities_by_name(model_path: str) -> dict[str, float]:
    """Return capability scores (0.0–1.0) based on filename keywords + known family baselines."""
    name = Path(model_path).stem.lower().replace("_", "-")

    # Try family baseline first
    scores: dict[str, float] = {}
    for family, baseline in FAMILY_BASELINES.items():
        if family in name:
            scores = dict(baseline)
            break

    # Overlay keyword boosts (keyword hit → set to 1.0 if not already higher)
    for cap, keywords in NAME_RULES.items():
        if any(kw in name for kw in keywords):
            scores[cap] = max(scores.get(cap, 0.0), 1.0)

    # Zero-fill missing caps
    for cap in CAPABILITIES:
        scores.setdefault(cap, 0.0)

    # If nothing at all detected, treat as generic chat model
    if not any(v > 0 for v in scores.values()):
        scores["chat"] = 0.5

    return scores


# ---------------------------------------------------------------------------
# Method 2: Mini-benchmark probe (requires running model)
# ---------------------------------------------------------------------------

# 5 tiny prompts per capability — designed to be answerable in < 100 tokens.
# We score by checking if the response contains expected patterns.  # xray: ignore[QUAL-014]

PROBE_PROMPTS: dict[str, list[dict[str, str]]] = {
    "chat": [
        {"prompt": "Hello! How are you today?", "expect": "hello|hi|good|great|fine|well"},
        {"prompt": "What's a good recipe for scrambled eggs?", "expect": "egg|pan|butter|salt|cook"},
        {"prompt": "Tell me a fun fact about dolphins.", "expect": "dolphin|ocean|mammal|brain|swim"},
        {"prompt": "Can you recommend a movie for a rainy day?", "expect": "movie|film|watch|recommend"},
        {"prompt": "What should I name my new kitten?", "expect": "name|kitten|cat|cute"},
    ],
    "reasoning": [
        {"prompt": "If all roses are flowers and some flowers fade quickly, can we conclude all roses fade quickly? Explain.", "expect": "cannot|no|some|not necessarily|invalid"},
        {"prompt": "A bat and a ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?", "expect": "0.05|5 cent|five cent"},
        {"prompt": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?", "expect": "5 minute"},
        {"prompt": "There are 3 boxes. One has apples, one has oranges, one has both. Labels are ALL wrong. You pick one fruit from the 'both' box and it's an apple. What's in each box?", "expect": "apple|orange"},  # xray: ignore[QUAL-013]
        {"prompt": "Is the statement 'This statement is false' true or false? Explain the paradox.", "expect": "paradox|liar|self-referent|neither|contradict"},
    ],
    "coding": [
        {"prompt": "Write a Python function to check if a string is a palindrome.", "expect": "def |return|==|reverse|[::-1]"},
        {"prompt": "What does 'git rebase' do?", "expect": "rebase|commit|branch|base|history"},
        {"prompt": "Fix this Python: def add(a, b) return a + b", "expect": ":|colon|def add"},
        {"prompt": "Write a SQL query to find the top 5 customers by total order amount.", "expect": "SELECT|ORDER BY|LIMIT|GROUP BY|SUM"},
        {"prompt": "Explain what a closure is in JavaScript.", "expect": "closure|function|scope|variable|inner"},
    ],
    "math": [
        {"prompt": "What is the derivative of x^3 + 2x?", "expect": "3x\\^?2|3x²|3x2|\\+ ?2"},
        {"prompt": "Solve for x: 2x + 5 = 13", "expect": "x ?= ?4|x ?is ?4|4"},
        {"prompt": "What is the integral of sin(x)?", "expect": "-cos|\\-cos"},
        {"prompt": "A circle has radius 5. What is its area?", "expect": "78\\.5|25π|25pi|25\\*pi"},
        {"prompt": "What is the probability of rolling two 6s with two dice?", "expect": "1/36|0\\.027|2\\.7"},
    ],
    "translation": [
        {"prompt": "Translate to French: 'The weather is beautiful today.'", "expect": "temps|beau|aujourd"},
        {"prompt": "Translate to Spanish: 'Where is the library?'", "expect": "dónde|biblioteca|donde"},
        {"prompt": "Translate to German: 'Good morning, how are you?'", "expect": "Guten Morgen|Wie geht"},
        {"prompt": "Translate to Japanese: 'Thank you very much.'", "expect": "ありがとう|domo|arigato"},
        {"prompt": "What language is this: 'Ciao, come stai?'", "expect": "Italian|italiano"},
    ],
    "creative": [
        {"prompt": "Write a haiku about autumn.", "expect": "leaf|leave|fall|autumn|wind|gold|red"},
        {"prompt": "Start a short story with: 'The old lighthouse keeper saw something strange in the fog.'", "expect": "lighthouse|fog|strange|light|sea|ocean"},
        {"prompt": "Write a limerick about a cat.", "expect": "cat|sat|mat|hat|fat"},
        {"prompt": "Describe a sunset using only colors and emotions.", "expect": "gold|red|orange|warm|peace|calm|purple"},
        {"prompt": "Write a 2-line poem about rain.", "expect": "rain|drop|fall|pour|cloud|wet"},
    ],
}


def _score_response(response: str, expect_pattern: str) -> float:
    """Score 0.0-1.0 based on whether response matches expected patterns."""
    if not response or len(response.strip()) < 5:
        return 0.0

    patterns = expect_pattern.split("|")
    matches = sum(1 for p in patterns if re.search(p, response, re.IGNORECASE))
    # Need at least 1 match, score scales with coverage
    if matches == 0:
        return 0.0
    return min(1.0, matches / max(len(patterns) * 0.3, 1))


def probe_teacher(query_fn, teacher: dict, capabilities: list[str] | None = None) -> dict[str, float]:
    """Run mini-benchmark probes and return capability scores (0-1 scale).

    Args:
        query_fn: callable(teacher, prompt, max_tokens, temperature) -> str
        teacher: dict with at least 'name' and 'gguf' keys
        capabilities: list of capabilities to probe (None = all)
    """
    caps_to_test = capabilities or list(PROBE_PROMPTS.keys())
    scores: dict[str, float] = {}

    for cap in CAPABILITIES:
        if cap not in caps_to_test or cap not in PROBE_PROMPTS:
            scores[cap] = 0.0
            continue

        prompts = PROBE_PROMPTS[cap]
        cap_scores: list[float] = []

        for item in prompts:
            try:
                response = query_fn(teacher, item["prompt"], max_tokens=150, temperature=0.3)
                score = _score_response(response, item["expect"])
                cap_scores.append(score)
            except Exception:  # xray: ignore[QUAL-011]
                cap_scores.append(0.0)

        scores[cap] = round(sum(cap_scores) / len(cap_scores), 3) if cap_scores else 0.0

    return scores


# ---------------------------------------------------------------------------
# Prompt classification — route prompts to best teachers
# ---------------------------------------------------------------------------

PROMPT_CATEGORY_PATTERNS: dict[str, list[str]] = {
    "coding": [
        r"\b(code|program|function|class|variable|debug|fix.*bug|implement|algorithm|API|endpoint)\b",
        r"\b(python|javascript|java|rust|sql|html|css|typescript|golang|c\+\+)\b",
        r"\b(git|docker|kubernetes|CI/CD|deploy|compile|import|library|framework)\b",
    ],
    "math": [
        r"\b(solve|equation|integral|derivative|calculus|algebra|probability|matrix)\b",
        r"\b(calculate|compute|proof|theorem|formula|percentage|fraction|ratio)\b",
        r"\b(math|geometric|trigonometric|logarithm|exponent|polynomial)\b",
    ],
    "reasoning": [
        r"\b(explain why|reason|logic|paradox|deduc|induc|analogy|syllogism)\b",
        r"\b(if.*then|conclude|infer|assume|therefore|contradiction|premise)\b",
        r"\b(think step by step|let's reason|analyze|compare and contrast)\b",
    ],
    "translation": [
        r"\b(translat|tradui|übersetze|翻译)\b",
        r"\b(in (french|spanish|german|chinese|japanese|korean|arabic|russian|portuguese|italian))\b",
        r"\b(what language|which language|言語)\b",
    ],
    "creative": [
        r"\b(write a (poem|story|haiku|limerick|song|script|dialogue))\b",
        r"\b(creative|imagine|fiction|narrative|metaphor|rhyme)\b",
        r"\b(describe.*using|paint.*words|make up)\b",
    ],
    "safety": [
        r"\b(safe|danger|harm|illegal|ethic|privacy|consent|bias|discriminat)\b",
        r"\b(should I|is it okay|appropriate|offensive|sensitive)\b",
    ],
    "ocr": [
        r"\b(image|picture|photo|screenshot|scan|OCR|read.*text|extract.*text)\b",
        r"\b(what does.*show|describe.*image|vision)\b",
    ],
}


def classify_prompt(prompt: str) -> str:
    """Classify a prompt into a capability category. Returns 'chat' as fallback."""
    text = prompt.lower()
    best_cap = "chat"
    best_score = 0

    for cap, patterns in PROMPT_CATEGORY_PATTERNS.items():
        score = sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))
        if score > best_score:
            best_score = score
            best_cap = cap

    return best_cap


def route_prompt_to_teachers(
    prompt: str,
    capability_matrix: dict[str, dict[str, float]],
    threshold: float = 0.3,
    min_teachers: int = 1,
) -> list[str]:
    """Pick which teachers should answer this prompt.

    Args:
        prompt: the prompt text
        capability_matrix: {teacher_name: {capability: score}}
        threshold: minimum score to qualify
        min_teachers: always include at least this many (pick top even if below threshold)

    Returns:
        list of teacher names that should answer this prompt
    """
    cap = classify_prompt(prompt)

    # Score teachers for this capability
    teacher_scores = [
        (name, scores.get(cap, 0.0))
        for name, scores in capability_matrix.items()  # xray: ignore[QUAL-005]
    ]
    teacher_scores.sort(key=lambda x: x[1], reverse=True)

    # Pick those above threshold
    qualified = [name for name, score in teacher_scores if score >= threshold]

    # Ensure minimum count
    if len(qualified) < min_teachers:
        qualified = [name for name, _ in teacher_scores[:min_teachers]]

    # For majority vote we want odd count
    if len(qualified) > 1 and len(qualified) % 2 == 0:
        qualified = qualified[:-1]  # drop the weakest

    return qualified


# ---------------------------------------------------------------------------
# Full profiler — combines name heuristic + optional probe
# ---------------------------------------------------------------------------

def profile_all_teachers(
    teacher_paths: list[str],
    query_fn=None,
    probe_caps: list[str] | None = None,
) -> dict:
    """Profile all teachers and return a full capability manifest.

    Returns:
        {
            "teachers": [
                {
                    "name": "deepseek-r1-8b-q8_0",
                    "gguf": "C:/AI/Models/deepseek-r1-8b-q8_0.gguf",
                    "capabilities": {"chat": 0.7, "reasoning": 0.9, ...},
                    "top_capabilities": ["reasoning", "chat"],
                    "method": "probe" | "name_heuristic"
                }
            ],
            "capability_matrix": {"deepseek-r1": {"chat": 0.7, ...}},
            "routing_ready": true
        }
    """
    teachers = []
    capability_matrix: dict[str, dict[str, float]] = {}

    for path in teacher_paths:
        name = Path(path).stem
        teacher = {"name": name, "gguf": path}

        # Always start with name heuristic
        name_scores = infer_capabilities_by_name(path)

        if query_fn is not None:
            # Probe with actual model
            probe_scores = probe_teacher(query_fn, teacher, probe_caps)
            # Merge: probe wins where it ran, name fills gaps
            final_scores = {cap: max(name_scores.get(cap, 0), probe_scores.get(cap, 0)) for cap in CAPABILITIES}
            method = "probe"
        else:
            final_scores = name_scores
            method = "name_heuristic"

        # Ensure all capabilities have a score
        for cap in CAPABILITIES:
            final_scores.setdefault(cap, 0.0)

        # Top capabilities = those above 0.3
        top_caps = sorted(
            [cap for cap, score in final_scores.items() if score >= 0.3],
            key=lambda c: final_scores[c],
            reverse=True,
        )

        teachers.append({
            "name": name,
            "gguf": path,
            "capabilities": final_scores,
            "top_capabilities": top_caps or ["chat"],
            "method": method,
        })
        capability_matrix[name] = final_scores

    return {
        "teachers": teachers,
        "capability_matrix": capability_matrix,
        "routing_ready": True,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Profile teacher model capabilities.")
    parser.add_argument("--teachers", nargs="+", required=True, help="GGUF paths for teacher models.")
    parser.add_argument("--probe", action="store_true", help="Run mini-benchmark probes (needs backend).")
    parser.add_argument("--backend", default="inprocess", choices=["inprocess", "server"], help="Backend for probing.")
    parser.add_argument("--output", default="", help="Save profile JSON to this path.")
    parser.add_argument("--probe-caps", nargs="*", help="Only probe these capabilities.")
    args = parser.parse_args()

    query_fn = None
    if args.probe:
        # Import backend from multi_teacher_generate
        sys.path.insert(0, str(SCRIPTS_DIR))
        from multi_teacher_generate import _make_inprocess_backend, _make_server_backend

        manifest = {"backend": args.backend, "teachers": [{"name": Path(p).stem, "gguf": p} for p in args.teachers]}
        if args.backend == "server":
            query_fn = _make_server_backend(manifest)
        else:
            query_fn = _make_inprocess_backend(manifest)

    profile = profile_all_teachers(args.teachers, query_fn, args.probe_caps)

    # Print summary
    print("\n=== Teacher Capability Profile ===\n")  # xray: ignore[PY-004]
    for t_info in profile["teachers"]:
        print(f"  {t_info['name']}  ({t_info['method']})")  # xray: ignore[PY-004]
        for cap in CAPABILITIES:
            score = t_info["capabilities"].get(cap, 0)
            bar = "#" * int(score * 10) + "." * (10 - int(score * 10))
            marker = " *" if cap in t_info["top_capabilities"] else ""
            print(f"    {cap:>12s}  {bar} {score:.2f}{marker}")  # xray: ignore[PY-004]
        print()  # xray: ignore[PY-004]

    # Save if requested
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)
        print(f"Saved to: {out_path}")  # xray: ignore[PY-004]

    return 0


SCRIPTS_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    raise SystemExit(main())
