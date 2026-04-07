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

r"""University of Distillation — Wizard UI.

Dean Zena (Gemma 4 GGUF) oversees a 3-phase distillation pipeline:
  Phase 1  Enrollment  — discover & enroll teacher GGUFs, credential check, cross-exam
  Phase 2  Teaching    — multi-teacher generation → hallucination gates → GOLD/SILVER/DROP
  Phase 3  Graduation  — generate training configs → train the student model

Left panel:  Zena chat (always visible) — chatbot + command interface
Right panel: 3 tabs (one phase per tab, only current phase active)

Launch:  python scripts/distill_ui.py
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import gradio as gr  # xray: ignore[SEC-015]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parent
PYTHON = sys.executable

# ---------------------------------------------------------------------------
# CSS — university-themed dark/light
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
:root {
    --zen-accent: #7c3aed;
    --zen-accent-hover: #6d28d9;
    --zen-success: #16a34a;
    --zen-warn: #d97706;
    --zen-danger: #dc2626;
    --zen-gold: #ca8a04;
    --zen-silver: #64748b;
}
.zen-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 20px; margin-bottom: 8px;
    border-bottom: 2px solid var(--zen-accent);
}
.zen-logo {
    font-size: 1.5rem; font-weight: 700; letter-spacing: -0.02em;
    background: linear-gradient(135deg, var(--zen-accent), #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.zen-subtitle {
    opacity: 0.6; font-size: 0.85rem;
}
.zen-phase-title {
    font-size: 1.15rem; font-weight: 600; margin: 8px 0 4px 0;
    color: var(--zen-accent);
}
/* Badge chips */
.zen-badge {
    display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: 0.8rem; font-weight: 600; margin-right: 4px;
}
.zen-badge-gold   { background: #fef3c7; color: #92400e; }
.zen-badge-silver { background: #f1f5f9; color: #334155; }
.zen-badge-drop   { background: #fee2e2; color: #991b1b; }
.zen-badge-pass   { background: #dcfce7; color: #166534; }
.zen-badge-flag   { background: #fff7ed; color: #9a3412; }
.zen-badge-fail   { background: #fee2e2; color: #991b1b; }
/* Stat cards */
.zen-stat-card {
    padding: 16px; border-radius: 10px; text-align: center;
    border: 1px solid var(--border-color-primary, #e5e7eb);
}
.zen-stat-card h3 { margin: 0 0 4px 0; font-size: 2rem; }
.zen-stat-card p  { margin: 0; opacity: 0.7; font-size: 0.85rem; }
.zen-gold   h3 { color: var(--zen-gold); }
.zen-silver h3 { color: var(--zen-silver); }
.zen-drop   h3 { color: var(--zen-danger); }
/* Chat styling */
.zena-chat { min-height: 400px; }
"""


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

def _run_script(cmd: list[str], timeout: int = 7200) -> tuple[int, str, str]:
    """Run a subprocess and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, cwd=str(ROOT_DIR),
    )
    return result.returncode, result.stdout, result.stderr


def _safe_prompt_count(prompts_path: str) -> int:
    """Count prompts in JSONL or JSON-array prompt files."""
    p = Path(prompts_path)
    if not p.is_file():
        return 0

    try:
        raw = p.read_text(encoding="utf-8").strip()
        if not raw:
            return 0
        if raw.startswith("["):
            rows = json.loads(raw)
            return len(rows) if isinstance(rows, list) else 0
        return sum(1 for line in raw.splitlines() if line.strip())
    except Exception:  # xray: ignore[QUAL-011]
        return 0


def _sanitize_teacher_name(name: str) -> str:
    """Match checkpoint naming in multi_teacher_generate.py."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def _format_eta(seconds: float) -> str:
    if seconds <= 0:
        return "0m"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    if h > 0:
        return f"{h}h {m}m"
    return f"{m}m"


def get_phase2_progress(teachers_text: str, prompts_path: str, output_dir: str) -> str:
    """Return live generation progress + ETA from checkpoint files."""
    teachers = [line.strip() for line in teachers_text.splitlines() if line.strip()]
    if not teachers:
        return "### Live Progress\n\nNo teachers configured yet."

    prompt_count = _safe_prompt_count(prompts_path)
    if prompt_count <= 0:
        return f"### Live Progress\n\nPrompts file not found or empty: `{prompts_path}`"

    ckpt_dir = Path(output_dir) / "checkpoints"
    if not ckpt_dir.is_dir():
        return (
            "### Live Progress\n\n"
            "No checkpoint directory yet. Start teaching first, then click refresh.\n\n"
            f"Expected: `{ckpt_dir}`"
        )

    rows = []
    elapsed_samples: list[float] = []
    total_done = 0
    total_target = prompt_count * len(teachers)

    for teacher_path in teachers:
        t_name = Path(teacher_path).stem
        ckpt_file = ckpt_dir / f"{_sanitize_teacher_name(t_name)}.jsonl"
        done = 0
        last_update = "-"

        if ckpt_file.is_file():
            with ckpt_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    done += 1
                    try:
                        rec = json.loads(line)
                        sec = rec.get("response", {}).get("elapsed_s", None)
                        if isinstance(sec, (int, float)) and sec > 0:
                            elapsed_samples.append(float(sec))
                    except Exception:  # xray: ignore[QUAL-002, QUAL-011]
                        pass
            try:
                ts = ckpt_file.stat().st_mtime
                age_s = max(0, int(time.time() - ts))
                last_update = f"{age_s}s ago"
            except OSError:  # xray: ignore[QUAL-002]
                pass

        done = min(done, prompt_count)
        pct = (100.0 * done / prompt_count) if prompt_count else 0.0
        total_done += done
        rows.append(f"| {t_name} | {done}/{prompt_count} | {pct:.1f}% | {last_update} |")

    overall_pct = (100.0 * total_done / total_target) if total_target else 0.0
    remaining = max(0, total_target - total_done)

    # Robust ETA: trim top 10% outliers from elapsed samples
    eta_text = "estimating..."
    if elapsed_samples:
        samples = sorted(elapsed_samples)
        keep = max(1, int(len(samples) * 0.9))
        trimmed = samples[:keep]
        avg_s = sum(trimmed) / len(trimmed)
        eta_text = f"{_format_eta(remaining * avg_s)} (avg {avg_s:.1f}s/response)"

    md = [
        "### Live Progress",
        "",
        f"**Overall:** {total_done}/{total_target} responses ({overall_pct:.1f}%)",
        f"**ETA to generation complete:** {eta_text}",
        "",
        "| Teacher | Progress | % | Last update |",
        "|---|---:|---:|---:|",
        *rows,
        "",
        f"Checkpoints: `{ckpt_dir}`",
    ]
    return "\n".join(md)


def _write_manifest(teachers_text: str, backend: str, output_dir: str) -> str:
    """Create a teacher_manifest.json from the textarea lines."""
    lines = [l.strip() for l in teachers_text.strip().splitlines() if l.strip()]
    if not lines:
        raise ValueError("No teacher paths provided.")
    teachers = []
    for line in lines:
        name = Path(line).stem
        teachers.append({"name": name, "gguf": line})
    manifest = {"backend": backend, "teachers": teachers}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "teacher_manifest.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return str(path)


def discover_gguf_models() -> list[dict]:
    """Scan common directories for .gguf files."""
    dirs = [
        "C:/AI/Models",
        os.path.expanduser("~/AI/Models"),
        os.path.expanduser("~/.cache/llama.cpp"),
    ]
    found = []
    seen: set[str] = set()
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
            found.append({"path": path, "name": Path(path).stem, "size_gb": round(size_gb, 1)})
    found.sort(key=lambda x: x["name"].lower())
    return found


# ---------------------------------------------------------------------------
# Lazy-loaded ZenaDean singleton
# ---------------------------------------------------------------------------
_dean = None


def _get_dean(zena_gguf: str):
    """Get or create the ZenaDean singleton."""
    global _dean  # xray: ignore[PY-006]
    if _dean is None or _dean.zena_gguf != zena_gguf:
        from zena_dean import ZenaDean
        _dean = ZenaDean(zena_gguf)
    return _dean


# ---------------------------------------------------------------------------
# Zena chat — command interface + LLM fallback
# ---------------------------------------------------------------------------

def _extract_gguf_paths(text: str) -> list[str]:
    """Extract any .gguf file paths from text."""
    paths = []
    for line in text.splitlines():
        line = line.strip()
        if line.lower().endswith(".gguf") and (line.startswith("/") or ":" in line[:3]):
            paths.append(line)
    return paths


def _zena_respond(message: str, chat_history: list, zena_gguf: str, teacher_list_text: str):
    """Handle Zena chat messages: commands + LLM fallback.

    Commands:
        scan / discover  — list available GGUFs
        enroll / add     — enroll teachers from teacher list or message paths
        cross-examine    — run peer review between enrolled teachers
        status           — show enrollment summary
        recommend <goal> — ask Zena who to enroll
        *anything else*  — LLM chat with Zena
    """
    msg_lower = message.strip().lower()

    # --- Command: scan / discover ---
    if msg_lower in ("scan", "discover", "find models", "list models"):
        models = discover_gguf_models()
        if not models:
            reply = "No .gguf models found in the default directories (C:/AI/Models, ~/AI/Models)."
        else:
            lines = [f"Found **{len(models)}** GGUF model(s):\n"]
            for m in models[:30]:
                lines.append(f"- `{m['path']}` ({m['size_gb']} GB)")
            if len(models) > 30:
                lines.append(f"\n… and {len(models) - 30} more.")
            reply = "\n".join(lines)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": reply})
        return chat_history, chat_history

    # --- Command: enroll / add ---
    if msg_lower.startswith(("enroll", "add")):
        if not zena_gguf.strip():
            reply = "Please set the **Zena GGUF path** first (top of chat panel)."
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": reply})
            return chat_history, chat_history

        dean = _get_dean(zena_gguf)
        # Get paths from the teacher list box or from the message itself
        paths = _extract_gguf_paths(teacher_list_text) if teacher_list_text.strip() else _extract_gguf_paths(message)
        if not paths:
            paths = [l.strip() for l in teacher_list_text.strip().splitlines() if l.strip()]

        if not paths:
            reply = "No teacher paths found. Paste GGUF paths into the **Teacher GGUFs** box in the Enrollment tab."
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": reply})
            return chat_history, chat_history

        enrolled = []
        for p in paths:
            if not os.path.isfile(p):
                enrolled.append(f"- ~~{Path(p).stem}~~ — file not found")
                continue
            info = dean.enroll_teacher(p)
            caps = ", ".join(info["top_capabilities"][:3])
            enrolled.append(f"- **{Path(p).stem}** — {caps}")

        reply = f"Enrolled {len([e for e in enrolled if '~~' not in e])} professor(s):\n" + "\n".join(enrolled)
        reply += "\n\n" + dean.get_enrollment_summary()
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": reply})
        return chat_history, chat_history

    # --- Command: cross-examine ---
    if msg_lower in ("cross-examine", "cross examine", "exam", "peer review"):
        if not zena_gguf.strip():
            reply = "Set the Zena GGUF path first."
        else:
            dean = _get_dean(zena_gguf)
            if len(dean.teachers) < 2:
                reply = "Need at least **2 enrolled professors** for cross-examination. Use `enroll` first."
            else:
                reply = "Running cross-examination — each professor evaluates the others… ⏳\n\n*(This takes a few minutes per teacher pair.)*"
                chat_history.append({"role": "user", "content": message})
                chat_history.append({"role": "assistant", "content": reply})
                # Run the actual cross-exam (blocking)
                dean.cross_examine()
                matrix_md = dean.get_peer_matrix_markdown()
                chat_history.append({"role": "assistant", "content": matrix_md})
                return chat_history, chat_history

        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": reply})
        return chat_history, chat_history

    # --- Command: status ---
    if msg_lower in ("status", "who", "enrolled", "professors"):
        if not zena_gguf.strip():
            reply = "Set the Zena GGUF path first."
        else:
            dean = _get_dean(zena_gguf)
            reply = dean.get_enrollment_summary()
            if dean.peer_matrix:
                reply += "\n\n" + dean.get_peer_matrix_markdown()
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": reply})
        return chat_history, chat_history

    # --- Command: recommend ---
    if msg_lower.startswith("recommend"):
        goal = message[len("recommend"):].strip() or "general distillation"
        if not zena_gguf.strip():
            reply = "Set the Zena GGUF path first."
        else:
            dean = _get_dean(zena_gguf)
            reply = dean.recommend_teachers(goal)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": reply})
        return chat_history, chat_history

    # --- Fallback: LLM chat ---
    if not zena_gguf.strip():
        reply = (
            "Hello! I'm **Zena**, Dean of the University of Distillation.\n\n"
            "To get started, set my **GGUF model path** at the top, then try:\n"
            "- `scan` — find available models\n"
            "- `enroll` — add teachers\n"
            "- `cross-examine` — professors evaluate each other\n"
            "- `status` — see who's enrolled\n"
            "- `recommend <goal>` — get model suggestions\n"
            "- Or just ask me anything!"
        )
    else:
        dean = _get_dean(zena_gguf)
        reply = dean.chat(message)

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": reply})
    return chat_history, chat_history


# ---------------------------------------------------------------------------
# Phase 1 — Enrollment helpers
# ---------------------------------------------------------------------------

def _profile_from_text(teachers_text: str) -> str:
    """Quick name-based enrollment summary without ZenaDean."""
    lines = [l.strip() for l in teachers_text.strip().splitlines() if l.strip()]
    if not lines:
        return "No teachers listed."
    try:
        sys.path.insert(0, str(SCRIPTS_DIR))
        from teacher_profiler import infer_capabilities_by_name
        rows = []
        for p in lines:
            caps = infer_capabilities_by_name(p)
            top = sorted([c for c, s in caps.items() if s >= 0.3], key=lambda c: caps[c], reverse=True)[:3]
            rows.append(f"- **{Path(p).stem}** — {', '.join(top) or 'general'}")
        return "**Quick credential check (name heuristic):**\n" + "\n".join(rows)
    except Exception as e:  # xray: ignore[QUAL-011]
        return f"Could not profile: {e}"


def on_credential_check(teachers_text, backend, output_dir, do_probe, zena_gguf):
    """Run credential check — profile teachers + optionally enroll with Zena."""
    if not teachers_text.strip():
        return "**Error:** no teacher GGUFs listed.", ""

    lines = [l.strip() for l in teachers_text.strip().splitlines() if l.strip()]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    profile_path = str(out / "teacher_profile.json")

    cmd = [
        PYTHON, str(SCRIPTS_DIR / "teacher_profiler.py"),
        "--teachers", *lines,
        "--backend", backend,
        "--output", profile_path,
    ]
    if do_probe:
        cmd.append("--probe")

    rc, stdout, stderr = _run_script(cmd, timeout=600)
    if rc != 0:
        err = stderr[-2000:] if stderr else "unknown"
        return f"**Credential check failed** (exit {rc})\n```\n{err}\n```", ""

    # Parse the profile
    try:
        with open(profile_path, encoding="utf-8") as f:
            profile = json.load(f)
        matrix = profile.get("capability_matrix", {})
        if not matrix:
            return "**Credential check done** (no capability data).", profile_path

        all_caps = sorted({c for scores in matrix.values() for c in scores})
        hdr = "| Professor | " + " | ".join(c.title() for c in all_caps) + " |"
        sep = "|---| " + " | ".join("---" for _ in all_caps) + " |"
        rows = []
        for name, scores in matrix.items():
            short = Path(name).stem if "/" in name or "\\" in name else name
            cells = []
            for c in all_caps:
                v = scores.get(c, 0.0)
                if v >= 0.7:
                    cells.append(f"**{v:.0%}** ✓")
                elif v >= 0.3:
                    cells.append(f"{v:.0%}")
                else:
                    cells.append(f"~~{v:.0%}~~")
            rows.append(f"| {short} | " + " | ".join(cells) + " |")

        md = "### Professor Expertise Matrix\n\n" + "\n".join([hdr, sep] + rows)
        md += "\n\n*✓ = strong (≥70%), ~~strikethrough~~ = weak (<30%). Prompts routed to qualified professors.*"

        # Also enroll with Zena if available
        if zena_gguf and zena_gguf.strip():
            dean = _get_dean(zena_gguf)
            for p in lines:
                if os.path.isfile(p):
                    dean.enroll_teacher(p)

        return md, profile_path
    except Exception as e:  # xray: ignore[QUAL-011]
        return f"**Credential check done** but error parsing: {e}", profile_path


def on_cross_exam(zena_gguf):
    """Run cross-examination between enrolled teachers."""
    if not zena_gguf or not zena_gguf.strip():
        return "Set the **Zena GGUF path** first."

    dean = _get_dean(zena_gguf)
    if len(dean.teachers) < 2:
        return "Need at least **2 enrolled professors**. Run **Credential Check** first."

    dean.cross_examine()
    return dean.get_peer_matrix_markdown()


# ---------------------------------------------------------------------------
# Phase 2 — Teaching (generate + halluc gates + purify)
# ---------------------------------------------------------------------------

def run_phase2_full(
    teachers_text: str,
    prompts_path: str,
    backend: str,
    output_dir: str,
    max_tokens: int,
    temperature: float,
    answer_th: float,
    reason_th: float,
    enable_halluc: bool,
    zena_gguf: str,
    profile_path: str,
):
    """Full Phase 2 pipeline: generate → halluc gates → purify.

    Returns: (status_md, gold_n, silver_n, drop_n, halluc_report_md)
    """
    if not teachers_text.strip():
        return "**Error:** no teachers listed.", 0, 0, 0, ""
    if not prompts_path.strip() or not Path(prompts_path).is_file():
        return f"**Error:** prompts file not found: `{prompts_path}`", 0, 0, 0, ""

    try:
        manifest_path = _write_manifest(teachers_text, backend, output_dir)
    except ValueError as e:
        return f"**Error:** {e}", 0, 0, 0, ""

    out_jsonl = str(Path(output_dir) / "teacher_responses.jsonl")

    # --- Step 1: Multi-teacher generation ---
    cmd = [
        PYTHON, str(SCRIPTS_DIR / "multi_teacher_generate.py"),
        "--manifest", manifest_path,
        "--prompts", prompts_path,
        "--out", out_jsonl,
        "--max-tokens", str(int(max_tokens)),
        "--temperature", str(temperature),
    ]
    if profile_path and Path(profile_path).is_file():
        cmd.extend(["--profile", profile_path])

    rc, stdout, stderr = _run_script(cmd)
    if rc != 0:
        err = stderr[-2000:] if stderr else "unknown"
        return f"**Generation failed** (exit {rc})\n```\n{err}\n```", 0, 0, 0, ""

    # --- Step 2: Hallucination gates (optional) ---
    halluc_md = ""
    if enable_halluc:
        halluc_md = _run_halluc_gates(out_jsonl, zena_gguf, output_dir)

    # --- Step 3: Purification ---
    cmd_purify = [
        PYTHON, str(SCRIPTS_DIR / "purify_teacher_outputs.py"),
        "--input", out_jsonl,
        "--out-dir", output_dir,
        "--answer-threshold", str(answer_th),
        "--reason-threshold", str(reason_th),
    ]
    rc, stdout, stderr = _run_script(cmd_purify, timeout=600)
    if rc != 0:
        err = stderr[-2000:] if stderr else "unknown"
        return f"**Purification failed** (exit {rc})\n```\n{err}\n```", 0, 0, 0, halluc_md

    # Read report
    report_path = Path(output_dir) / "purification_report.json"
    gold_n = silver_n = drop_n = 0
    if report_path.is_file():
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)  # xray: ignore[PY-005]
        gold_n = report.get("gold_count", 0)
        silver_n = report.get("silver_count", 0)
        drop_n = report.get("dropped_count", 0)

    count = 0
    if Path(out_jsonl).is_file():
        with open(out_jsonl, encoding="utf-8") as f:
            count = sum(1 for l in f if l.strip())

    routing = " (smart routing)" if profile_path else ""
    status = (
        f"**Teaching complete** — {count} samples generated{routing}\n\n"
        f"Purified: "
        f'<span class="zen-badge zen-badge-gold">GOLD {gold_n}</span> '
        f'<span class="zen-badge zen-badge-silver">SILVER {silver_n}</span> '
        f'<span class="zen-badge zen-badge-drop">DROP {drop_n}</span>\n\n'
        f"Output: `{output_dir}`"
    )
    return status, gold_n, silver_n, drop_n, halluc_md


def _run_halluc_gates(responses_jsonl: str, zena_gguf: str, output_dir: str) -> str:
    """Run hallucination gates on the teacher responses.

    Returns a markdown summary.
    """
    try:
        from hallucination_gates import run_hallucination_pipeline, GateVerdict
    except ImportError:  # xray: ignore[QUAL-002]
        sys.path.insert(0, str(SCRIPTS_DIR))
        from hallucination_gates import run_hallucination_pipeline, GateVerdict

    dean = _get_dean(zena_gguf) if zena_gguf and zena_gguf.strip() else None

    total = 0
    passed = 0
    flagged = 0
    failed = 0
    all_reports = []

    with open(responses_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:  # xray: ignore[PY-005]
                            row = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                            row = {}
            prompt = row.get("prompt", "")
            teachers_data = row.get("teachers", {})
            teacher_answers = {}
            for tname, tdata in teachers_data.items():
                teacher_answers[tname] = tdata.get("answer", tdata.get("raw", ""))

            if not teacher_answers:
                continue

            reports = run_hallucination_pipeline(
                prompt=prompt,
                teacher_answers=teacher_answers,
                zena_dean=dean,
                borderline_only=True,
            )

            for name, report in reports.items():
                total += 1
                if report.final_verdict == GateVerdict.PASS:
                    passed += 1
                elif report.final_verdict == GateVerdict.FLAG:
                    flagged += 1
                else:
                    failed += 1
                all_reports.append(report)

    # Save detailed report
    report_out = Path(output_dir) / "hallucination_report.json"
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "teacher": r.teacher,
                    "prompt_id": r.prompt_id,
                    "verdict": r.final_verdict.value,
                    "score": r.final_score,
                    "gates": [
                        {"gate": g.gate, "verdict": g.verdict.value, "score": g.score}
                        for g in r.gate_results
                    ],
                }
                for r in all_reports
            ],
            f, indent=2,
        )

    md = (
        f"### Hallucination Gate Report\n\n"
        f"Scanned **{total}** teacher responses:\n\n"
        f'<span class="zen-badge zen-badge-pass">PASS {passed}</span> '
        f'<span class="zen-badge zen-badge-flag">FLAG {flagged}</span> '
        f'<span class="zen-badge zen-badge-fail">FAIL {failed}</span>\n\n'
    )
    if failed > 0:
        md += f"⚠️ **{failed}** responses failed hallucination checks and will be demoted in purification.\n"
    md += f"\nDetailed report: `{report_out}`"
    return md


# ---------------------------------------------------------------------------
# Phase 3 — Graduation (configs + training)
# ---------------------------------------------------------------------------

def run_phase3_gen_configs(output_dir: str, student_model: str, cpu_safe: bool):
    """Generate SFT/DPO training configs."""
    if not student_model.strip():
        return "**Error:** specify a student model."

    config_out = str(ROOT_DIR / "examples" / "distillation" / "auto")
    cmd = [
        PYTHON, str(SCRIPTS_DIR / "gen_distill_configs.py"),
        "--student", student_model,
        "--data-dir", output_dir,
        "--out-dir", config_out,
        "--tag", "distill_auto",
    ]
    if cpu_safe:
        cmd.append("--cpu-safe")

    rc, stdout, stderr = _run_script(cmd, timeout=60)
    if rc != 0:
        err = stderr[-2000:] if stderr else "unknown"
        return f"**Config generation failed**\n```\n{err}\n```"

    yamls = sorted(glob.glob(os.path.join(config_out, "*.yaml")))
    lines = ["**Configs generated:**\n"]
    for y in yamls:
        lines.append(f"- `{y}`")
    lines.append(f"\nRun manually: `llamafactory-cli train {yamls[0] if yamls else config_out}`")
    return "\n".join(lines)


def run_phase3_train(output_dir: str, student_model: str, cpu_safe: bool):
    """Run SFT + DPO training."""
    config_out = ROOT_DIR / "examples" / "distillation" / "auto"
    sft_yaml = config_out / "distill_auto_sft.yaml"
    dpo_yaml = config_out / "distill_auto_dpo.yaml"

    if not sft_yaml.is_file():
        return "**Error:** SFT config not found. Click **Generate Training Configs** first."

    lines = ["**Training started…**\n"]

    # SFT
    cmd_sft = ["llamafactory-cli", "train", str(sft_yaml)]
    rc, stdout, stderr = _run_script(cmd_sft, timeout=14400)
    if rc != 0:
        return f"**SFT failed** (exit {rc})\n```\n{stderr[-2000:]}\n```"
    lines.append("✅ SFT training **complete**\n")

    # DPO
    if dpo_yaml.is_file():
        lines.append("Running DPO training…\n")
        cmd_dpo = ["llamafactory-cli", "train", str(dpo_yaml)]
        rc, stdout, stderr = _run_script(cmd_dpo, timeout=14400)
        if rc != 0:
            lines.append(f"DPO **failed** (exit {rc})\n```\n{stderr[-1500:]}\n```")
        else:
            lines.append("✅ DPO training **complete**\n")

    lines.append("\n**🎓 Graduation complete.** Check `saves/distill_auto/` for outputs.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build the UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.violet,
        secondary_hue=gr.themes.colors.gray,
    )

    with gr.Blocks(
        title="University of Distillation",
        theme=theme,
        css=CUSTOM_CSS,
    ) as app:

        # ── Header ──────────────────────────────────────────
        gr.HTML(
            '<div class="zen-header">'
            '  <span class="zen-logo">🎓 University of Distillation</span>'
            '  <span class="zen-subtitle">Dean Zena × ZEN Forge</span>'
            "</div>"
        )

        # ── Hidden state ────────────────────────────────────
        profile_path_state = gr.State("")
        _welcome_msg = {
            "role": "assistant",
            "content": (
                "Welcome to the **University of Distillation**! "
                "I'm **Zena**, your Dean (Gemma 4).\n\n"
                "**3 professors pre-enrolled:**\n"
                "- 🧠 **deepseek-r1** — reasoning specialist\n"
                "- 💬 **qwen2.5-instruct** — chat specialist\n"
                "- 💻 **qwen2.5-coder** — coding & chat\n\n"
                "Try: `scan` · `enroll` · `cross-examine` · `status` · "
                "`recommend <goal>` · or just ask me anything!"
            ),
        }
        chat_state = gr.State([_welcome_msg])

        with gr.Row():

            # ============================================================
            # LEFT PANEL — Zena Chat (always visible)
            # ============================================================
            with gr.Column(scale=2):
                gr.HTML('<div class="zen-phase-title">💬 Dean Zena</div>')
  # xray: ignore-next[PORT-002]
                zena_gguf = gr.Textbox(  # xray: ignore[PORT-002]
                    label="Zena GGUF (Gemma 4)",
                    value="C:/AI/Models/google_gemma-4-26B-A4B-it-Q4_K_M.gguf",  # xray: ignore[PORT-002]
                    placeholder="C:/AI/Models/google_gemma-4-26B-A4B-it-Q4_K_M.gguf",  # xray: ignore[PORT-002]
                    info="Path to the Gemma 4 GGUF that powers Dean Zena. 12.1 tok/s · 8192 ctx.",
                )

                chatbot = gr.Chatbot(
                    label="Chat with Zena",
                    elem_classes=["zena-chat"],
                    type="messages",
                    height=500,
                    value=[
                        {
                            "role": "assistant",
                            "content": (
                                "Welcome to the **University of Distillation**! "
                                "I'm **Zena**, your Dean (Gemma 4 26B).\n\n"
                                "**Zena_007 — 3 professors enrolled:**\n"
                                "- 🌍 **Gemma 4 26B A4B** — multilingual, chat, creative\n"
                                "- 💬 **Qwen2.5 14B Instruct** — chat, translation\n"
                                "- 🌐 **Mistral 7B v0.3** — European languages, chat\n\n"
                                "**629 prompts ready:** 491 translation · 28 OCR · 110 chat\n"
                                "9 languages: EN · RO · HU · HE · FR · ES · DE · ID · PT\n\n"
                                "Try: `scan` · `enroll` · `cross-examine` · `status` · "
                                "`recommend <goal>` · or just ask me anything!"
                            ),
                        }
                    ],
                )

                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder="Type a command or ask Zena anything…",
                        show_label=False,
                        scale=5,
                    )
                    chat_send = gr.Button("Send", variant="primary", scale=1)

                gr.Markdown(
                    "*Commands:* `scan` · `enroll` · `cross-examine` · "
                    "`status` · `recommend <goal>` · or free chat",
                    elem_classes=["zen-subtitle"],
                )

            # ============================================================
            # RIGHT PANEL — 3-Phase Tabs
            # ============================================================
            with gr.Column(scale=3):
                with gr.Tabs():

                    # ── Phase 1: Enrollment ─────────────────────
                    with gr.Tab("Phase 1 · Enrollment"):
                        gr.HTML('<div class="zen-phase-title">📋 Teacher Enrollment</div>')

                        teachers_box = gr.Textbox(
                            lines=6,
                            label="Teacher GGUFs — one per line (odd count = majority vote)",
                            info=(
                                "Skills auto-detected: chat · reasoning · coding · math · "
                                "translation · ocr · safety · creative"
                            ),
                            value=(
                                "C:/AI/Models/google_gemma-4-26B-A4B-it-Q4_K_M.gguf\n"
                                "C:/AI/Models/Qwen2.5-14B-Instruct-Q4_K_M.gguf\n"
                                "C:/AI/Models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"
                            ),
                        )

                        with gr.Row():
                            prompts_path = gr.Textbox(
                                label="Prompts file",
                                value="data/zena007_prompts.jsonl",
                                placeholder="data/distill_prompts.jsonl",
                                info="JSONL, JSON (alpaca), or messages format. 629 Zena_007 prompts.",
                                scale=3,
                            )
                            backend_dd = gr.Dropdown(
                                choices=["inprocess", "server"],
                                value="inprocess",
                                label="Backend",
                                info="inprocess = direct GGUF via zen_core_libs",
                                scale=1,
                            )

                        output_dir = gr.Textbox(
                            value="data/purified",
                            label="Output directory",
                        )

                        with gr.Row():
                            profile_btn = gr.Button(
                                "🔍 Credential Check", variant="secondary", size="lg", scale=2,
                            )
                            probe_check = gr.Checkbox(
                                label="Deep probe (5 prompts × 7 caps)",
                                value=False,
                                info="Slower but more accurate than name heuristic.",
                                scale=1,
                            )
                            cross_exam_btn = gr.Button(
                                "⚔️ Cross-Examine", variant="secondary", size="lg", scale=2,
                            )

                        enrollment_status = gr.Markdown(
                            value=(
                                "**Quick credential check (name heuristic):**\n"
                                "- **Gemma 4 26B A4B** — 🌍 multilingual, 💬 chat, ✨ creative\n"
                                "- **Qwen2.5 14B Instruct** — 💬 chat, 🌍 translation\n"
                                "- **Mistral 7B v0.3** — 💬 chat, 🌍 European languages\n\n"
                                "*3 professors enrolled (odd = majority vote ready). "
                                "Click **Credential Check** for deep probe.*"
                            ),
                        )
                        credential_report = gr.Markdown(value="")
                        peer_matrix_display = gr.Markdown(value="")

                    # ── Phase 2: Teaching ───────────────────────
                    with gr.Tab("Phase 2 · Teaching"):
                        gr.HTML('<div class="zen-phase-title">📚 Teaching Session</div>')

                        with gr.Accordion("⚙️ Teaching Settings", open=False):
                            with gr.Row():
                                max_tokens = gr.Slider(
                                    64, 4096, value=512, step=64,
                                    label="Max tokens per response",
                                )
                                temperature = gr.Slider(
                                    0.0, 2.0, value=0.7, step=0.05,
                                    label="Temperature",
                                )
                            with gr.Row():
                                answer_th = gr.Slider(
                                    0.5, 1.0, value=0.85, step=0.01,
                                    label="Answer consensus threshold",
                                    info="N-gram similarity for teachers to 'agree'.",
                                )
                                reason_th = gr.Slider(
                                    0.2, 1.0, value=0.60, step=0.01,
                                    label="Reasoning alignment threshold",
                                    info="Below this → SILVER (DPO pair).",
                                )
                            enable_halluc = gr.Checkbox(
                                label="Enable hallucination gates (5-gate pipeline)",
                                value=True,
                                info="Runs consistency, drift, grounding, confidence & Zena judge before purification.",
                            )

                        start_teach_btn = gr.Button(
                            "🎓 Start Teaching", variant="primary", size="lg",
                        )

                        phase2_status = gr.Markdown(
                            value=(
                                "*3 professors enrolled: "
                                "🌍 Gemma 4 · Qwen2.5 · Mistral — "
                                "629 prompts (translation + OCR + multilingual chat). "
                                "Click **Start Teaching** when ready.*"
                            ),
                        )

                        with gr.Row():
                            refresh_progress_btn = gr.Button("🔄 Refresh Progress", variant="secondary")
                        progress_display = gr.Markdown(
                            value=(
                                "### Live Progress\n\n"
                                "Click **Refresh Progress** to see teacher-by-teacher progress and ETA."
                            )
                        )

                        with gr.Row():
                            gold_display = gr.Number(label="🥇 GOLD (SFT)", value=0, interactive=False)
                            silver_display = gr.Number(label="🥈 SILVER (DPO)", value=0, interactive=False)
                            drop_display = gr.Number(label="❌ DROP", value=0, interactive=False)

                        halluc_report = gr.Markdown(value="")

                    # ── Phase 3: Graduation ─────────────────────
                    with gr.Tab("Phase 3 · Graduation"):
                        gr.HTML('<div class="zen-phase-title">🎓 Student Graduation</div>')

                        with gr.Row():
                            student_model = gr.Textbox(
                                value="Qwen/Qwen2.5-1.5B-Instruct",
                                label="Student model",
                                info="HuggingFace ID or local path. 1.5B params — fast training on consumer hardware.",
                                scale=3,
                            )
                            cpu_safe = gr.Checkbox(
                                value=True,
                                label="CPU-safe (no bf16)",
                                info="Enabled by default — safe for CPU-only training.",
                            )

                        with gr.Row():
                            gen_configs_btn = gr.Button(
                                "📄 Generate Training Configs", variant="secondary", size="lg",
                            )
                            start_train_btn = gr.Button(
                                "🚀 Start Training", variant="primary", size="lg",
                            )

                        phase3_status = gr.Markdown(
                            value=(
                                "*Student: **Qwen2.5-1.5B-Instruct** (1.5B params). "
                                "Complete Phase 2 to produce GOLD/SILVER data, then generate configs and train.*"
                            ),
                        )

        # ── Wiring ──────────────────────────────────────────────

        # Chat send
        def on_chat(msg, history, zena_path, teacher_text):
            if not msg.strip():
                return history, history, ""
            new_hist, state = _zena_respond(msg, history, zena_path, teacher_text)
            return new_hist, state, ""

        chat_send.click(
            fn=on_chat,
            inputs=[chat_input, chat_state, zena_gguf, teachers_box],
            outputs=[chatbot, chat_state, chat_input],
        )
        chat_input.submit(
            fn=on_chat,
            inputs=[chat_input, chat_state, zena_gguf, teachers_box],
            outputs=[chatbot, chat_state, chat_input],
        )

        # Quick enrollment preview on teacher text change
        teachers_box.change(
            fn=lambda txt: _profile_from_text(txt) if txt.strip() else "*Paste teacher GGUFs above.*",
            inputs=[teachers_box],
            outputs=[enrollment_status],
        )

        # Credential check
        profile_btn.click(
            fn=on_credential_check,
            inputs=[teachers_box, backend_dd, output_dir, probe_check, zena_gguf],
            outputs=[credential_report, profile_path_state],
        )

        # Cross-examination
        cross_exam_btn.click(
            fn=on_cross_exam,
            inputs=[zena_gguf],
            outputs=[peer_matrix_display],
        )

        # Start teaching (Phase 2)
        def on_start_teaching(
            teachers_text, prompts, backend, outdir, maxtok, temp,
            ans_th, reas_th, halluc_on, zena_path, prof_path,
        ):
            return run_phase2_full(
                teachers_text, prompts, backend, outdir,
                maxtok, temp, ans_th, reas_th,
                halluc_on, zena_path, prof_path,
            )

        start_teach_btn.click(
            fn=on_start_teaching,
            inputs=[
                teachers_box, prompts_path, backend_dd, output_dir,
                max_tokens, temperature, answer_th, reason_th,
                enable_halluc, zena_gguf, profile_path_state,
            ],
            outputs=[
                phase2_status, gold_display, silver_display, drop_display, halluc_report,
            ],
        )

        # Refresh Phase 2 live progress + ETA from checkpoint files
        refresh_progress_btn.click(
            fn=get_phase2_progress,
            inputs=[teachers_box, prompts_path, output_dir],
            outputs=[progress_display],
        )

        # Generate configs (Phase 3)
        gen_configs_btn.click(
            fn=run_phase3_gen_configs,
            inputs=[output_dir, student_model, cpu_safe],
            outputs=[phase3_status],
        )

        # Start training (Phase 3)
        start_train_btn.click(
            fn=run_phase3_train,
            inputs=[output_dir, student_model, cpu_safe],
            outputs=[phase3_status],
        )

    return app


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main():
    app = build_ui()
    app.queue().launch(
        server_name="0.0.0.0",
        server_port=7870,
        inbrowser=True,
        show_api=False,
    )


if __name__ == "__main__":
    main()
