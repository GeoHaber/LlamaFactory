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
# CSS — RAG-bench-inspired palette (clean, no global overrides)
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
/* ════════════════════════════════════════════════════════════
   University of Distillation — palette from RAG Test Bench
   Targeted classes only; Gradio handles its own inputs.
   ════════════════════════════════════════════════════════════ */

/* ── Top header bar (matches RAG bench .top) ───────────────── */
.zen-top {
    display: flex; align-items: center; gap: 12px;
    padding: 10px 20px; margin-bottom: 12px;
    background: var(--background-fill-primary, #fff);
    border: 1px solid var(--border-color-primary, #e5e7eb);
    border-radius: 10px;
    box-shadow: 0 1px 3px rgb(0 0 0/.08);
}
.zen-avatar {
    width: 36px; height: 36px; border-radius: 50%;
    background: hsl(221 83% 53%); color: #fff;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; font-weight: 800; flex-shrink: 0;
    border: 2px solid hsl(221 83% 53%/.25);
    user-select: none;
}
.zen-brand-title {
    font-size: 15px; font-weight: 700; letter-spacing: -.02em;
    color: var(--body-text-color, #111);
}
.zen-brand-title span { color: hsl(221 83% 52%); }
.zen-brand-sub {
    font-size: 10px; font-weight: 500; letter-spacing: .04em;
    text-transform: uppercase; color: var(--body-text-color-subdued, #6b7280);
}

/* Pipeline step indicators (like RAG bench phase nav) */
.zen-steps {
    display: flex; align-items: center; gap: 0;
    margin-left: auto; flex-shrink: 0;
}
.zen-step {
    display: flex; align-items: center; gap: 5px;
    padding: 3px 10px; font-size: 11px; font-weight: 600;
    color: var(--body-text-color-subdued, #6b7280);
    border-radius: 20px; transition: all .2s;
    white-space: nowrap;
}
.zen-step.active {
    background: hsl(221 83% 53%/.1);
    color: hsl(221 83% 45%);
}
.zen-step-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--border-color-primary, #d1d5db); flex-shrink: 0;
}
.zen-step.active .zen-step-dot {
    background: hsl(221 83% 53%);
    box-shadow: 0 0 6px hsl(221 83% 53%/.5);
}
.zen-step-sep {
    color: var(--border-color-primary, #9ca3af);
    font-size: 14px; padding: 0 2px;
}

/* ── Phase section labels ──────────────────────────────────── */
.zen-phase-title {
    font-size: 13px; font-weight: 700;
    color: var(--body-text-color, #111);
    margin: 4px 0 10px;
}
.zen-subtitle {
    font-size: 11px; color: var(--body-text-color-subdued, #6b7280);
}

/* ── Stat row (from RAG bench .p-stats / .p-stat) ─────────── */
.zen-stats-row { display: flex; gap: 12px; margin: 8px 0; }
.zen-stat {
    flex: 1; text-align: center; padding: 14px 8px;
    background: var(--background-fill-secondary, #f9fafb);
    border: 1px solid var(--border-color-primary, #e5e7eb);
    border-radius: 8px;
}
.zen-stat .val {
    display: block; font-size: 26px; font-weight: 700;
    color: hsl(221 83% 53%); line-height: 1.1;
}
.zen-stat .lbl {
    font-size: 10px; color: var(--body-text-color-subdued, #6b7280);
    text-transform: uppercase; letter-spacing: .5px; margin-top: 2px;
}
.zen-stat.gold   .val { color: hsl(38 85% 36%); }
.zen-stat.silver .val { color: hsl(220 10% 45%); }
.zen-stat.drop   .val { color: hsl(0 72% 44%); }

/* ── Status dot line (from RAG bench .status-line) ────────── */
.zen-status-line {
    font-size: 11px; margin-top: 6px;
    display: flex; align-items: center; gap: 5px;
    color: var(--body-text-color-subdued, #6b7280);
}
.zen-status-dot {
    width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0;
    background: var(--border-color-primary, #d1d5db);
}
.zen-status-dot.ok   { background: hsl(142 71% 42%); }
.zen-status-dot.warn { background: hsl(38 92% 50%); }
.zen-status-dot.err  { background: hsl(0 84% 58%); }

/* ── Chip (from RAG bench .chip) ──────────────────────────── */
.zen-chip {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 2px 10px;
    background: var(--background-fill-secondary, #f3f4f6);
    border: 1px solid var(--border-color-primary, #e5e7eb);
    border-radius: 20px; font-size: 11px;
    color: var(--body-text-color-subdued, #6b7280); flex-shrink: 0;
}
.zen-chip b { color: hsl(142 71% 36%); font-weight: 700; }

/* ── Badge chips ──────────────────────────────────────────── */
.zen-badge {
    display: inline-block; padding: 2px 9px; border-radius: 12px;
    font-size: 0.78rem; font-weight: 600; margin-right: 4px;
}
.zen-badge-gold   { background: hsl(38 92% 50%/.15);  color: hsl(38 70% 28%); }
.zen-badge-silver { background: hsl(220 10% 50%/.15); color: hsl(220 10% 30%); }
.zen-badge-drop   { background: hsl(0 84% 60%/.12);   color: hsl(0 60% 36%); }
.zen-badge-pass   { background: hsl(142 71% 45%/.15); color: hsl(142 50% 24%); }
.zen-badge-flag   { background: hsl(38 92% 50%/.15);  color: hsl(38 70% 28%); }
.zen-badge-fail   { background: hsl(0 84% 60%/.12);   color: hsl(0 60% 36%); }

/* ── Scrollbars ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-color-primary, #d1d5db); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--body-text-color-subdued, #6b7280); }

@media (max-width: 920px) {
    .zen-top { padding: 8px 12px; }
    .zen-brand-title { font-size: 13px; }
    .zen-steps { display: none; }
}
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


def _run_script_live(cmd: list[str], timeout: int = 7200):
    """Start a subprocess in the background; return the Popen object for live polling."""
    return subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=str(ROOT_DIR),
    )


def _collect_proc(proc, timeout: int = 7200) -> tuple[int, str, str]:
    """Wait for a Popen to finish and collect stdout/stderr."""
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
    return proc.returncode, out or "", err or ""



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
        return '<p style="color:var(--error-text-color,#dc2626)">No teacher GGUFs listed.</p>', ""

    lines = [l.strip() for l in teachers_text.strip().splitlines() if l.strip()]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    profile_path = str(out / "teacher_profile.json")

    # ── Fast in-process name-heuristic (always runs — no subprocess, no GGUF load) ──
    probe_banner = ""
    try:
        sys.path.insert(0, str(SCRIPTS_DIR))
        from teacher_profiler import profile_all_teachers
        profile_obj = profile_all_teachers(lines, query_fn=None, probe_caps=None)
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile_obj, f, indent=2, ensure_ascii=False)
    except Exception as e:  # xray: ignore[QUAL-011]
        return f'<p style="color:var(--error-text-color,#dc2626)">Name-heuristic profile failed: {e}</p>', ""

    if do_probe:
        # Launch deep probe as a DETACHED background process — never block the UI.
        # Loading 3 large GGUFs + running inference takes 30-120 minutes.
        # Results are written to profile_path when done; re-run credential check to see them.
        cmd = [
            PYTHON, str(SCRIPTS_DIR / "teacher_profiler.py"),
            "--teachers", *lines,
            "--backend", backend,
            "--output", profile_path,
            "--probe",
        ]
        try:
            log_path = Path(output_dir) / "deep_probe.log"
            with open(log_path, "w", encoding="utf-8") as log_fh:
                subprocess.Popen(
                    cmd,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    cwd=str(ROOT_DIR),
                    close_fds=True,
                )
            probe_banner = (
                '<div style="margin-top:10px;padding:10px 14px;border-radius:6px;'
                'background:hsl(38,92%,45%,0.15);border:1px solid hsl(38,92%,55%);'
                'font-size:11px;color:hsl(38,92%,65%);">'
                '<b>Deep probe running in background</b> (may take 30-120 min per GGUF).<br>'
                f'Progress: <code>{log_path}</code>&nbsp; — re-run Credential Check when done to refresh cards.'
                "</div>"
            )
        except Exception as exc:  # xray: ignore[QUAL-011]
            probe_banner = (
                f'<p style="color:var(--error-text-color,#dc2626)">Could not start background probe: {exc}</p>'
            )

    # ── Parse the profile and build HTML cards ──
    try:
        with open(profile_path, encoding="utf-8") as f:
            profile = json.load(f)
        matrix = profile.get("capability_matrix", {})
        if not matrix:
            return '<p>Credential check done (no capability data).</p>', profile_path

        all_caps = sorted({c for scores in matrix.values() for c in scores})

        # Capability colour palette -- matches RAG bench pipe-card colours
        CAP_COLORS: dict[str, tuple[str, str]] = {
            "chat":        ("hsl(221,83%,65%)",  "hsl(221,83%,53%,0.18)"),
            "reasoning":   ("hsl(271,81%,70%)",  "hsl(271,81%,56%,0.18)"),
            "coding":      ("hsl(142,71%,55%)",  "hsl(142,71%,38%,0.18)"),
            "math":        ("hsl(38,92%,60%)",   "hsl(38,92%,45%,0.18)"),
            "translation": ("hsl(187,85%,55%)",  "hsl(187,85%,38%,0.18)"),
            "ocr":         ("hsl(0,72%,65%)",    "hsl(0,72%,51%,0.18)"),
            "safety":      ("hsl(160,60%,55%)",  "hsl(160,60%,38%,0.18)"),
            "creative":    ("hsl(328,75%,67%)",  "hsl(328,75%,52%,0.18)"),
        }

        def _bar(score: float, color: str) -> str:
            pct = int(score * 100)
            return (
                f'<div style="flex:1;height:5px;border-radius:3px;'
                f'background:var(--border-color-primary,#374151);">'
                f'<div style="height:100%;width:{pct}%;border-radius:3px;background:{color};"></div></div>'
            )

        def _pill(cap: str, score: float) -> str:
            color, bg = CAP_COLORS.get(cap, ("hsl(220,10%,60%)", "hsl(220,10%,50%,0.15)"))
            weight = "700" if score >= 0.7 else "500"
            return (
                f'<span style="display:inline-flex;align-items:center;gap:3px;'
                f'padding:2px 7px;border-radius:20px;font-size:10px;font-weight:{weight};'
                f'background:{bg};color:{color};border:1px solid {color};margin:2px 2px 2px 0;">'
                f'<span style="width:7px;height:7px;border-radius:50%;background:{color};'
                f'display:inline-block;"></span>'
                f'{cap}&nbsp;{score:.0%}</span>'
            )

        # Build one card per professor
        cards = (
            '<div style="font-size:11px;font-weight:700;text-transform:uppercase;'
            'letter-spacing:.5px;color:var(--body-text-color-subdued,#9ca3af);margin:4px 0 10px;">'
            'Professor Expertise Matrix</div>'
            '<div style="display:flex;flex-direction:column;gap:10px;">'
        )
        for name, scores in matrix.items():
            short = Path(name).stem if ("/" in name or "\\" in name) else name
            top = sorted(
                [(c, scores.get(c, 0.0)) for c in all_caps],
                key=lambda x: x[1], reverse=True,
            )
            pills = "".join(_pill(cap, sc) for cap, sc in top if sc >= 0.1)
            bars = ""
            for cap, sc in top:
                color, _ = CAP_COLORS.get(cap, ("hsl(220,10%,60%)", ""))
                bars += (
                    f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">'
                    f'<span style="font-size:9px;width:66px;text-align:right;'
                    f'color:var(--body-text-color-subdued,#9ca3af);flex-shrink:0;">{cap}</span>'
                    + _bar(sc, color)
                    + f'<span style="font-size:9px;width:28px;font-weight:600;'
                    f'color:var(--body-text-color,#f0f0f0);">{sc:.0%}</span></div>'
                )
            # Family icon letter (first letter of model family)
            icon_letter = short[0].upper() if short else "P"
            cards += (
                f'<div style="padding:14px;'
                f'background:var(--background-fill-secondary,#1f2937);'
                f'border:1px solid var(--border-color-primary,#374151);border-radius:8px;">'
                f'<div style="display:flex;align-items:flex-start;gap:10px;">'
                f'<div style="width:36px;height:36px;border-radius:8px;'
                f'background:hsl(221,83%,53%,0.20);color:hsl(221,83%,70%);'
                f'display:flex;align-items:center;justify-content:center;'
                f'font-size:16px;font-weight:800;flex-shrink:0;">{icon_letter}</div>'
                f'<div style="flex:1;min-width:0;">'
                f'<div style="font-size:12px;font-weight:700;'
                f'color:var(--body-text-color,#f0f0f0);margin-bottom:5px;'
                f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{short}</div>'
                f'<div style="display:flex;flex-wrap:wrap;">{pills}</div>'
                f'</div>'
                f'<div style="min-width:170px;flex-shrink:0;">{bars}</div>'
                f'</div></div>'
            )
        cards += (
            '</div>'
            '<div style="margin-top:8px;font-size:10px;'
            'color:var(--body-text-color-subdued,#9ca3af);">'
            'Filled pill = capable (&gt;=70%). '
            'Enable <b>Deep probe</b> for live inference scores.</div>'
        )

        # Enroll with Zena dean (does not require files to exist on disk)
        if zena_gguf and zena_gguf.strip():
            try:
                dean = _get_dean(zena_gguf)
                for p in lines:
                    dean.enroll_teacher(p)
            except Exception:  # xray: ignore[QUAL-011]
                pass  # enrollment is best-effort; cross-exam will re-try

        return cards + probe_banner, profile_path
    except Exception as e:  # xray: ignore[QUAL-011]
        return f'<p style="color:var(--error-text-color,#dc2626)">Profile parse error: {e}</p>', profile_path


def on_cross_exam(zena_gguf):
    """Run cross-examination between enrolled teachers."""
    if not zena_gguf or not zena_gguf.strip():
        return "Set the **Zena GGUF path** (top of Phase 1) first."

    dean = _get_dean(zena_gguf)
    if len(dean.teachers) < 2:
        return (
            "**3 professors are listed** but not yet enrolled with Zena Dean.\n\n"
            "Click **Credential Check** above to enroll them, then retry Cross-Examine."
        )

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
        from hallucination_gates import GateVerdict, run_hallucination_pipeline
    except ImportError:  # xray: ignore[QUAL-002]
        sys.path.insert(0, str(SCRIPTS_DIR))
        from hallucination_gates import GateVerdict, run_hallucination_pipeline

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

    lines.append("\n**Graduation complete.** Check `saves/distill_auto/` for outputs.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Full pipeline generator -- runs all 9 stages with live UI updates
# ---------------------------------------------------------------------------

_PIPELINE_STAGES = [
    ("generate",  "Stage 1 -- Expert Teaching   (multi-teacher generation)"),
    ("halluc",    "Stage 2 -- Hallucination Gates"),
    ("purify",    "Stage 3 -- Purification   (GOLD / SILVER / DROP)"),
    ("configs",   "Stage 4 -- Training Config Generation"),
    ("sft",       "Stage 5 -- SFT Training"),
    ("dpo",       "Stage 6 -- DPO Training"),
    ("merge",     "Stage 7 -- LoRA Merge"),
    ("gguf",      "Stage 8 -- GGUF Export"),
    ("eval",      "Stage 9 -- Graduation Exam   (student vs teachers)"),
    ("dashboard", "Stage 10 -- Graduation Report"),
]

_STAGE_ICONS: dict[str, str] = {
    "wait": "[ ]",
    "running": "[>]",
    "done": "[+]",
    "skip": "[~]",
    "fail": "[!]",
}
_STAGE_COLORS: dict[str, str] = {
    "wait":    "hsl(220,10%,45%)",
    "running": "hsl(38,92%,55%)",
    "done":    "hsl(142,71%,50%)",
    "skip":    "hsl(220,10%,55%)",
    "fail":    "hsl(0,72%,58%)",
}


def _pipeline_html(stage_status: dict[str, str], current_msg: str = "") -> str:
    """Render the pipeline stage log as HTML."""
    rows: list[str] = []
    current_sid = next((s for s, st in stage_status.items() if st == "running"), None)
    for sid, slabel in _PIPELINE_STAGES:
        st = stage_status.get(sid, "wait")
        icon = _STAGE_ICONS[st]
        color = _STAGE_COLORS[st]
        bg = "var(--background-fill-secondary,#1f2937)" if st != "wait" else "transparent"
        border = "var(--border-color-primary,#374151)" if st != "wait" else "transparent"
        msg_html = (
            f'<div style="font-size:9px;color:var(--body-text-color-subdued,#9ca3af);'
            f'margin-top:3px;padding-left:22px;">{current_msg}</div>'
            if sid == current_sid and current_msg
            else ""
        )
        rows.append(
            f'<div style="padding:7px 12px;border-radius:6px;margin-bottom:3px;'
            f'background:{bg};border:1px solid {border};">'
            f'<span style="font-family:monospace;font-size:11px;margin-right:8px;">{icon}</span>'
            f'<span style="font-size:11px;font-weight:600;color:{color};">{slabel}</span>'
            f'{msg_html}</div>'
        )
    return (
        '<div style="font-size:11px;font-weight:700;text-transform:uppercase;'
        'letter-spacing:.5px;color:var(--body-text-color-subdued,#9ca3af);'
        'margin-bottom:8px;">Full Pipeline</div>'
        '<div>' + "".join(rows) + "</div>"
    )



def _extract_eta_snippet(progress_md: str) -> str:
    """Pull the first ETA/progress line from get_phase2_progress() markdown output."""
    for line in progress_md.splitlines():
        if "ETA" in line or "%" in line:
            return re.sub(r"\*\*(.+?)\*\*", r"\1", line).strip()
    return ""


def run_full_pipeline_gen(
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
    do_train: bool,
    student_model: str,
    cpu_safe: bool,
    saves_tag: str,
):
    """Generator: run all pipeline stages with live UI updates.

    Yields: (pipeline_log_html, gold_n, silver_n, drop_n, phase3_status_md, exam_html)
    """
    ss: dict[str, str] = {s: "wait" for s, _ in _PIPELINE_STAGES}
    gold_n = silver_n = drop_n = 0
    exam_html = ""

    def _emit(msg: str = "", p3: str = "", exam: str = "") -> tuple:
        return _pipeline_html(ss, msg), gold_n, silver_n, drop_n, p3, exam

    if not teachers_text.strip():
        ss["generate"] = "fail"
        yield _emit("No teachers listed.")
        return
    if not prompts_path.strip() or not Path(prompts_path).is_file():
        ss["generate"] = "fail"
        yield _emit(f"Prompts file not found: {prompts_path}")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_jsonl = str(out / "teacher_responses.jsonl")
    tag = saves_tag.strip() or "distill_auto"
    saves_dir = ROOT_DIR / "saves" / tag
    saves_dir.mkdir(parents=True, exist_ok=True)
    config_out = ROOT_DIR / "examples" / "distillation" / "auto"
    n_teachers = len([l.strip() for l in teachers_text.strip().splitlines() if l.strip()])

    # ── Stage 1: Generate ────────────────────────────────────────────────────
    def _responses_sufficient() -> bool:
        if not Path(out_jsonl).is_file():
            return False
        n_prompts = _safe_prompt_count(prompts_path)
        if n_prompts == 0:
            return False
        try:
            n_rows = sum(1 for l in open(out_jsonl, encoding="utf-8") if l.strip())
        except Exception:
            return False
        return n_rows >= n_teachers * n_prompts * 0.5

    if _responses_sufficient():
        ss["generate"] = "skip"
        yield _emit("teacher_responses.jsonl already complete -- skipping generation.")
    else:
        ss["generate"] = "running"
        yield _emit("Starting expert teachers...")
        try:
            manifest_path = _write_manifest(teachers_text, backend, output_dir)
        except ValueError as exc:
            ss["generate"] = "fail"
            yield _emit(str(exc))
            return

        cmd_gen = [
            PYTHON,
            str(SCRIPTS_DIR / "multi_teacher_generate.py"),
            "--manifest", manifest_path,
            "--prompts", prompts_path,
            "--out", out_jsonl,
            "--max-tokens", str(int(max_tokens)),
            "--temperature", str(temperature),
        ]
        if profile_path and Path(profile_path).is_file():
            cmd_gen.extend(["--profile", profile_path])

        proc_gen = _run_script_live(cmd_gen)
        while proc_gen.poll() is None:
            eta = _extract_eta_snippet(
                get_phase2_progress(teachers_text, prompts_path, output_dir)
            )
            yield _emit(eta or "Generating responses...")
            time.sleep(6)

        rc, _, err = _collect_proc(proc_gen)
        if rc != 0:
            ss["generate"] = "fail"
            yield _emit(f"Generation failed (exit {rc}): {err[-300:].strip()}")
            return
        ss["generate"] = "done"
        yield _emit("Generation complete.")

    # ── Stage 2: Hallucination gates ─────────────────────────────────────────
    if enable_halluc:
        halluc_rpt = out / "hallucination_report.json"
        if halluc_rpt.is_file():
            ss["halluc"] = "skip"
            yield _emit("Hallucination report already exists -- skipping.")
        else:
            ss["halluc"] = "running"
            yield _emit("Running 5-gate hallucination pipeline...")
            _run_halluc_gates(out_jsonl, zena_gguf, output_dir)
            ss["halluc"] = "done"
            yield _emit("Hallucination gates complete.")
    else:
        ss["halluc"] = "skip"
        yield _emit("Hallucination gates disabled.")

    # ── Stage 3: Purify ──────────────────────────────────────────────────────
    purify_rpt = out / "purification_report.json"
    if purify_rpt.is_file():
        ss["purify"] = "skip"
        try:
            with open(purify_rpt, encoding="utf-8") as fh:
                rpt = json.load(fh)
            gold_n = rpt.get("gold_count", 0)
            silver_n = rpt.get("silver_count", 0)
            drop_n = rpt.get("dropped_count", 0)
        except Exception:
            pass
        yield _emit(f"Purification already done -- GOLD {gold_n} / SILVER {silver_n} / DROP {drop_n}")
    else:
        ss["purify"] = "running"
        yield _emit("Purifying teacher responses...")
        cmd_purify = [
            PYTHON, str(SCRIPTS_DIR / "purify_teacher_outputs.py"),
            "--input", out_jsonl,
            "--out-dir", str(out),
            "--answer-threshold", str(answer_th),
            "--reason-threshold", str(reason_th),
        ]
        rc, _, err = _collect_proc(_run_script_live(cmd_purify), timeout=600)
        if rc != 0:
            ss["purify"] = "fail"
            yield _emit(f"Purification failed: {err[-300:].strip()}")
            return
        if purify_rpt.is_file():
            try:
                with open(purify_rpt, encoding="utf-8") as fh:
                    rpt = json.load(fh)
                gold_n = rpt.get("gold_count", 0)
                silver_n = rpt.get("silver_count", 0)
                drop_n = rpt.get("dropped_count", 0)
            except Exception:
                pass
        ss["purify"] = "done"
        yield _emit(f"GOLD {gold_n}  SILVER {silver_n}  DROP {drop_n}")

    if not do_train:
        for sid, _ in _PIPELINE_STAGES:
            if ss[sid] == "wait":
                ss[sid] = "skip"
        yield _emit("Training skipped (unchecked).")
        return

    # ── Stage 4: Generate configs ─────────────────────────────────────────────
    sft_yaml = config_out / "distill_auto_sft.yaml"
    dpo_yaml = config_out / "distill_auto_dpo.yaml"
    if sft_yaml.is_file():
        ss["configs"] = "skip"
        yield _emit("Training configs already exist -- skipping config gen.")
    else:
        ss["configs"] = "running"
        yield _emit(f"Generating configs for {student_model}...")
        cmd_cfg = [
            PYTHON, str(SCRIPTS_DIR / "gen_distill_configs.py"),
            "--student", student_model,
            "--data-dir", str(out),
            "--out-dir", str(config_out),
            "--tag", "distill_auto",
        ]
        if cpu_safe:
            cmd_cfg.append("--cpu-safe")
        rc, _, err = _collect_proc(_run_script_live(cmd_cfg), timeout=120)
        if rc != 0:
            ss["configs"] = "fail"
            yield _emit(f"Config gen failed: {err[-300:].strip()}")
            return
        ss["configs"] = "done"
        yield _emit("Training configs ready.")

    # ── Stage 5: SFT ─────────────────────────────────────────────────────────
    sft_flag = saves_dir / "sft_complete.flag"
    if sft_flag.is_file():
        ss["sft"] = "skip"
        yield _emit("SFT already complete -- skipping.", "SFT skipped (flag).")
    else:
        ss["sft"] = "running"
        yield _emit("SFT training started (may take 20-120 min)...", "SFT training...")
        proc_sft = _run_script_live(
            ["llamafactory-cli", "train", str(sft_yaml)], timeout=86400
        )
        while proc_sft.poll() is None:
            loss_hint = ""
            for tl in sorted((saves_dir).glob("**/trainer_log.jsonl")):
                try:
                    rows = [json.loads(l) for l in open(tl, encoding="utf-8") if l.strip()]
                    if rows:
                        last = rows[-1]
                        loss_hint = (
                            f"loss {last.get('loss', '?'):.4f}  "
                            f"step {last.get('current_steps','?')}/{last.get('total_steps','?')}"
                        )
                except Exception:
                    pass
            yield _emit(loss_hint or "SFT training...", f"SFT: {loss_hint or 'training...'}")
            time.sleep(15)
        rc, _, err = _collect_proc(proc_sft)
        if rc != 0:
            ss["sft"] = "fail"
            yield _emit(f"SFT failed: {err[-300:].strip()}", f"SFT FAILED: {err[-200:]}")
            return
        sft_flag.touch()
        ss["sft"] = "done"
        yield _emit("SFT complete.", "SFT complete.")

    # ── Stage 6: DPO ─────────────────────────────────────────────────────────
    dpo_flag = saves_dir / "dpo_complete.flag"
    if not dpo_yaml.is_file():
        ss["dpo"] = "skip"
        yield _emit("No DPO config (no SILVER data) -- skipping.")
    elif dpo_flag.is_file():
        ss["dpo"] = "skip"
        yield _emit("DPO already complete -- skipping.")
    else:
        ss["dpo"] = "running"
        yield _emit("DPO training started...", "DPO training...")
        proc_dpo = _run_script_live(
            ["llamafactory-cli", "train", str(dpo_yaml)], timeout=86400
        )
        while proc_dpo.poll() is None:
            yield _emit("DPO training...", "DPO training...")
            time.sleep(15)
        rc, _, err = _collect_proc(proc_dpo)
        if rc != 0:
            ss["dpo"] = "fail"
            yield _emit(f"DPO failed: {err[-300:].strip()}", f"DPO FAILED: {err[-200:]}")
            return
        dpo_flag.touch()
        ss["dpo"] = "done"
        yield _emit("DPO complete.", "DPO complete.")

    # ── Stage 7: Merge ────────────────────────────────────────────────────────
    merge_yaml = config_out / "distill_auto_merge.yaml"
    merged_dir = saves_dir / "merged"
    merge_config_json = merged_dir / "config.json"
    if merge_config_json.is_file():
        ss["merge"] = "skip"
        yield _emit("Merged model already exists -- skipping.")
    elif not merge_yaml.is_file():
        ss["merge"] = "skip"
        yield _emit("No merge config -- skipping.")
    else:
        ss["merge"] = "running"
        yield _emit("Merging LoRA adapters into base model...", "Merging...")
        rc, _, err = _collect_proc(
            _run_script_live(["llamafactory-cli", "export", str(merge_yaml)]), timeout=3600
        )
        if rc != 0:
            ss["merge"] = "fail"
            yield _emit(f"Merge failed: {err[-300:].strip()}", "Merge FAILED.")
            return
        ss["merge"] = "done"
        yield _emit("Merge complete.", "Merge complete.")

    # ── Stage 8: GGUF Export ──────────────────────────────────────────────────
    gguf_dir = saves_dir / "gguf"
    gguf_results = gguf_dir / "slim_down_results.jsonl"
    if gguf_results.is_file():
        ss["gguf"] = "skip"
        yield _emit("GGUF export already done -- skipping.")
    elif not merged_dir.is_dir():
        ss["gguf"] = "skip"
        yield _emit("No merged model -- skipping GGUF export.")
    else:
        ss["gguf"] = "running"
        yield _emit("Converting to GGUF Q4_K_M...", "GGUF export...")
        gguf_dir.mkdir(parents=True, exist_ok=True)
        cmd_gguf = [
            PYTHON, str(SCRIPTS_DIR / "slim_down.py"),
            "--model-dir", str(merged_dir),
            "--out-dir", str(gguf_dir),
            "--tag", tag,
            "--quant", "q4_k_m",
        ]
        rc, _, err = _collect_proc(_run_script_live(cmd_gguf), timeout=3600)
        if rc != 0:
            ss["gguf"] = "fail"
            yield _emit(f"GGUF export failed: {err[-300:].strip()}")
        else:
            ss["gguf"] = "done"
            yield _emit("GGUF export complete.")

    # ── Stage 9: Graduation Exam ──────────────────────────────────────────────
    scorecards = saves_dir / "eval_scorecards.jsonl"
    probes_candidates = [
        out / "eval_probes.jsonl",
        out / "purified" / "eval_probes.jsonl",
        Path(prompts_path),
    ]
    probes_path_p = next((p for p in probes_candidates if p.is_file()), None)

    if scorecards.is_file():
        ss["eval"] = "skip"
        yield _emit("Eval scorecards already exist -- skipping exam.")
    elif probes_path_p is None:
        ss["eval"] = "skip"
        yield _emit("No eval_probes.jsonl found -- skipping exam.")
    else:
        ss["eval"] = "running"
        yield _emit(f"Running graduation exam with {probes_path_p.name}...", "Examining students...")
        cmd_eval = [
            PYTHON, str(SCRIPTS_DIR / "eval_student_panel.py"),
            "--saves-tag", tag,
            "--probes", str(probes_path_p),
        ]
        proc_eval = _run_script_live(cmd_eval, timeout=7200)
        while proc_eval.poll() is None:
            yield _emit("Exam in progress...", "Exam running...")
            time.sleep(10)
        rc, _, err = _collect_proc(proc_eval)
        if rc != 0:
            ss["eval"] = "fail"
            exam_html = (
                '<p style="color:hsl(0,72%,58%);font-family:monospace;">'
                f"Exam failed (exit {rc}): {err[-400:]}</p>"
            )
            yield _emit("Exam failed.", exam=exam_html)
        else:
            ss["eval"] = "done"
            yield _emit("Exam complete.", "Exam complete.")

    # ── Stage 10: Graduation Dashboard ────────────────────────────────────────
    ss["dashboard"] = "running"
    yield _emit("Generating graduation report...")
    rpt_md = saves_dir / "graduation_report.md"
    cmd_dash = [
        PYTHON, str(SCRIPTS_DIR / "graduation_dashboard.py"),
        "--saves-tag", tag,
        "--export-markdown", str(rpt_md),
    ]
    rc, _, err = _collect_proc(_run_script_live(cmd_dash), timeout=120)
    if rc == 0 and rpt_md.is_file():
        ss["dashboard"] = "done"
        import html as _htmlmod
        raw = rpt_md.read_text(encoding="utf-8")
        exam_html = (
            '<div style="padding:12px;background:var(--background-fill-secondary,#1f2937);'
            'border:1px solid var(--border-color-primary,#374151);border-radius:8px;">'
            '<pre style="white-space:pre-wrap;font-size:11px;margin:0;'
            'color:var(--body-text-color,#f0f0f0);">'
            + _htmlmod.escape(raw)
            + "</pre></div>"
        )
    else:
        ss["dashboard"] = "fail"
        exam_html = exam_html or "<p>Dashboard generation failed. Check saves/ folder manually.</p>"

    final_p3_md = f"**Pipeline complete.** Outputs in `saves/{tag}/`."
    yield _pipeline_html(ss, "All done!"), gold_n, silver_n, drop_n, final_p3_md, exam_html


# ---------------------------------------------------------------------------
# Build the UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="University of Distillation",
        css=CUSTOM_CSS,
    ) as app:

        # ── Header (RAG-bench style: avatar + brand + phase steps) ──
        gr.HTML(
            '<div class="zen-top">'
            '  <div class="zen-avatar">Z</div>'
            '  <div>'
            '    <div class="zen-brand-title"><span>University</span> of Distillation</div>'
            '    <div class="zen-brand-sub">Dean Zena &times; ZEN Forge</div>'
            '  </div>'
            '  <div class="zen-steps">'
            '    <span class="zen-step active"><span class="zen-step-dot"></span>Enrollment</span>'
            '    <span class="zen-step-sep">&#8250;</span>'
            '    <span class="zen-step"><span class="zen-step-dot"></span>Teaching</span>'
            '    <span class="zen-step-sep">&#8250;</span>'
            '    <span class="zen-step"><span class="zen-step-dot"></span>Graduation</span>'
            '  </div>'
            '</div>'
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
                    elem_id="zena_gguf",
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
                                "**Zena_007 r2 — 3 professors enrolled:**\n"
                                "- **Gemma 4 26B A4B** — multilingual, chat, creative\n"
                                "- **Qwen2.5 14B Instruct** — chat, translation\n"
                                "- **Mistral 7B v0.3** — European languages, chat\n\n"
                                "**2 students enrolled (Forge Matrix):**\n"
                                "- **Zena_007_a** — Qwen2.5-0.5B, LoRA r8, SFT only\n"
                                "- **Zena_007_b** — Qwen2.5-3B, LoRA r16, SFT + DPO\n\n"
                                "**629 prompts ready:** 491 translation · 28 OCR · 110 chat\n\n"
                                "Try: `scan` · `enroll` · `cross-examine` · `status` · "
                                "`recommend <goal>` · or just ask me anything!"
                            ),
                        }
                    ],
                )

                with gr.Row():
                    chat_input = gr.Textbox(
                        elem_id="chat_input",
                        placeholder="Type a command or ask Zena anything…",
                        show_label=False,
                        scale=5,
                    )
                    chat_send = gr.Button("Send", variant="primary", scale=1, elem_id="chat_send")

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
                            elem_id="teachers_box",
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
                                elem_id="prompts_path",
                                label="Prompts file",
                                value="data/zena007_prompts.jsonl",
                                placeholder="data/distill_prompts.jsonl",
                                info="JSONL, JSON (alpaca), or messages format. 629 Zena_007 prompts.",
                                scale=3,
                            )
                            backend_dd = gr.Dropdown(
                                elem_id="backend_dd",
                                choices=["inprocess", "server"],
                                value="inprocess",
                                label="Backend",
                                info="inprocess = direct GGUF via zen_core_libs",
                                scale=1,
                            )

                        output_dir = gr.Textbox(
                            elem_id="output_dir",
                            value="data/purified",
                            label="Output directory",
                            info="Artifacts, manifests, checkpoints, and purified datasets are saved here.",
                        )

                        with gr.Row():
                            profile_btn = gr.Button(
                                "🔍 Credential Check", variant="secondary", size="lg", scale=2, elem_id="profile_btn",
                            )
                            probe_check = gr.Checkbox(
                                elem_id="probe_check",
                                label="Deep probe (5 prompts × 7 caps)",
                                value=False,
                                info="Slower but more accurate than name heuristic.",
                                scale=1,
                            )
                            cross_exam_btn = gr.Button(
                                "⚔️ Cross-Examine", variant="secondary", size="lg", scale=2, elem_id="cross_exam_btn",
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
                        credential_report = gr.HTML(value="")
                        peer_matrix_display = gr.Markdown(value="")

                    # ── Phase 2: Teaching ───────────────────────
                    with gr.Tab("Phase 2 · Teaching"):
                        gr.HTML('<div class="zen-phase-title">📚 Teaching Session</div>')

                        with gr.Accordion("⚙️ Teaching Settings", open=False):
                            with gr.Row():
                                max_tokens = gr.Slider(
                                    64, 4096, value=512, step=64,
                                    elem_id="max_tokens",
                                    label="Max tokens per response",
                                    info="Higher values capture longer reasoning but increase generation time.",
                                )
                                temperature = gr.Slider(
                                    0.0, 2.0, value=0.7, step=0.05,
                                    elem_id="temperature",
                                    label="Temperature",
                                    info="Lower is more deterministic, higher is more diverse.",
                                )
                            with gr.Row():
                                answer_th = gr.Slider(
                                    0.5, 1.0, value=0.85, step=0.01,
                                    elem_id="answer_th",
                                    label="Answer consensus threshold",
                                    info="N-gram similarity for teachers to 'agree'.",
                                )
                                reason_th = gr.Slider(
                                    0.2, 1.0, value=0.60, step=0.01,
                                    elem_id="reason_th",
                                    label="Reasoning alignment threshold",
                                    info="Below this → SILVER (DPO pair).",
                                )
                            enable_halluc = gr.Checkbox(
                                elem_id="enable_halluc",
                                label="Enable hallucination gates (5-gate pipeline)",
                                value=True,
                                info="Runs consistency, drift, grounding, confidence & Zena judge before purification.",
                            )

                        with gr.Row():
                            start_teach_btn = gr.Button(
                                "Start Teaching (Phase 2 only)",
                                variant="secondary", size="lg", elem_id="start_teach_btn",
                            )
                            run_all_btn = gr.Button(
                                "Run Full Pipeline (9 stages)",
                                variant="primary", size="lg", elem_id="run_all_btn",
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
                            refresh_progress_btn = gr.Button(
                                "🔄 Refresh Progress", variant="secondary", elem_id="refresh_progress_btn"
                            )
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
                        pipeline_log = gr.HTML(value="", label="Pipeline Progress")

                    # ── Phase 3: Graduation ─────────────────────
                    with gr.Tab("Phase 3 · Graduation"):
                        gr.HTML('<div class="zen-phase-title">Student Graduation</div>')

                        # Student A/B cards — Forge Matrix
                        gr.HTML(
                            '<div style="margin-bottom:12px;">'
                            '<div style="font-size:11px;font-weight:700;text-transform:uppercase;'
                            'letter-spacing:.5px;color:var(--body-text-color-subdued,#6b7280);'
                            'margin-bottom:8px;">Enrolled Students &mdash; Forge Matrix</div>'
                            '<div style="display:flex;gap:10px;">'
                            # — Card A —
                            '<div style="flex:1;padding:14px;'
                            'background:var(--background-fill-secondary,#f9fafb);'
                            'border:1px solid var(--border-color-primary,#e5e7eb);border-radius:8px;">'
                            '<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">'
                            '<div style="width:32px;height:32px;border-radius:8px;'
                            'background:hsl(221 83% 53%/.12);color:hsl(221 83% 45%);'
                            'display:flex;align-items:center;justify-content:center;'
                            'font-size:13px;font-weight:800;flex-shrink:0;">A</div>'
                            '<div>'
                            '<div style="font-size:13px;font-weight:700;'
                            'color:var(--body-text-color,#111);">Zena_007_a</div>'
                            '<div style="font-size:10px;color:var(--body-text-color-subdued,#6b7280);">'
                            'Qwen2.5-0.5B-Instruct</div>'
                            '</div></div>'
                            '<div style="display:flex;gap:4px;flex-wrap:wrap;">'
                            '<span class="zen-chip">LoRA r8</span>'
                            '<span class="zen-chip">lr&nbsp;2e-4</span>'
                            '<span class="zen-chip">2 epochs</span>'
                            '<span class="zen-chip">SFT only</span>'
                            '</div></div>'
                            # — Card B —
                            '<div style="flex:1;padding:14px;'
                            'background:var(--background-fill-secondary,#f9fafb);'
                            'border:1px solid var(--border-color-primary,#e5e7eb);border-radius:8px;">'
                            '<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">'
                            '<div style="width:32px;height:32px;border-radius:8px;'
                            'background:hsl(142 71% 45%/.12);color:hsl(142 60% 32%);'
                            'display:flex;align-items:center;justify-content:center;'
                            'font-size:13px;font-weight:800;flex-shrink:0;">B</div>'
                            '<div>'
                            '<div style="font-size:13px;font-weight:700;'
                            'color:var(--body-text-color,#111);">Zena_007_b</div>'
                            '<div style="font-size:10px;color:var(--body-text-color-subdued,#6b7280);">'
                            'Qwen2.5-3B-Instruct</div>'
                            '</div></div>'
                            '<div style="display:flex;gap:4px;flex-wrap:wrap;">'
                            '<span class="zen-chip">LoRA r16</span>'
                            '<span class="zen-chip">lr&nbsp;2e-5</span>'
                            '<span class="zen-chip">2 epochs</span>'
                            '<span class="zen-chip">SFT + DPO</span>'
                            '</div></div>'
                            '</div>'
                            '<div style="margin-top:6px;font-size:10px;'
                            'color:var(--body-text-color-subdued,#6b7280);">'
                            'Matrix: data/forge_matrix/zena007_two_new_students.yaml'
                            '</div></div>'
                        )

                        with gr.Row():
                            student_model = gr.Textbox(
                                elem_id="student_model",
                                value="Qwen/Qwen2.5-0.5B-Instruct",
                                label="Override student model (single run)",
                                info="Used only for single-student config gen. Forge Matrix above trains both A and B.",
                                scale=3,
                            )
                            cpu_safe = gr.Checkbox(
                                elem_id="cpu_safe",
                                value=True,
                                label="CPU-safe (no bf16)",
                                info="Safe for CPU-only training.",
                            )

                        with gr.Row():
                            saves_tag_box = gr.Textbox(
                                elem_id="saves_tag_box",
                                value="distill_auto",
                                label="Saves tag (subfolder under saves/)",
                                info="Outputs land in saves/<tag>/ — change per experiment.",
                                scale=2,
                            )
                            do_train_check = gr.Checkbox(
                                elem_id="do_train_check",
                                value=True,
                                label="Run training in full pipeline",
                                info="If unchecked, full pipeline stops after purification.",
                            )

                        with gr.Row():
                            gen_configs_btn = gr.Button(
                                "Generate Training Configs", variant="secondary", size="lg", elem_id="gen_configs_btn",
                            )
                            start_train_btn = gr.Button(
                                "Start Training", variant="primary", size="lg", elem_id="start_train_btn",
                            )

                        phase3_status = gr.Markdown(
                            value=(
                                "*2 students enrolled: **Zena_007_a** (0.5B, SFT only) and "
                                "**Zena_007_b** (3B, SFT + DPO). "
                                "Complete Phase 2 to produce GOLD/SILVER data, then launch the Forge Matrix.*"
                            ),
                        )

                    # ── Phase 4: Exam & Graduation ─────────────────────────
                    with gr.Tab("Phase 4 · Exam"):
                        gr.HTML('<div class="zen-phase-title">Graduation Exam</div>')
                        gr.Markdown(
                            "*Exam results appear here after the **Run Full Pipeline** button completes "
                            "all 9 stages. You can also run `eval_student_panel.py` manually.*"
                        )
                        exam_html_display = gr.HTML(
                            value="<p style='color:var(--body-text-color-subdued,#9ca3af);font-size:12px;'>"
                            "No exam data yet. Click <b>Run Full Pipeline</b> in Phase 2 to begin.</p>"
                        )

                gr.HTML(
                    """
                    <script>
                    (function () {
                        const tips = {
                            zena_gguf: "Dean model GGUF path used for chat, recommendations, and hallucination judging.",
                            chat_input: "Try commands: scan, enroll, cross-examine, status, recommend <goal>.",
                            chat_send: "Send message to Zena Dean.",
                            teachers_box: "Paste one teacher GGUF per line, odd teacher count is best for majority voting.",
                            prompts_path: "Prompt dataset used in Phase 2 generation.",
                            backend_dd: "inprocess runs local GGUFs directly, server uses an external llama-server.",
                            output_dir: "Folder for manifests, checkpoints, teacher outputs, and purified datasets.",
                            profile_btn: "Run capability profiling before routing prompts to teachers.",
                            probe_check: "Enables slower but deeper profiling probes.",
                            cross_exam_btn: "Have teachers review each other to surface weak experts.",
                            max_tokens: "Maximum tokens per teacher response.",
                            temperature: "Creativity and variation control.",
                            answer_th: "Agreement threshold for GOLD assignment.",
                            reason_th: "Reasoning-alignment threshold used for SILVER pairs.",
                            enable_halluc: "Run hallucination gate checks before purification.",
                            start_teach_btn: "Run generate, gates, and purify end-to-end (Phase 2 only).",
                            run_all_btn: "Run the full 9-stage pipeline: generate → hallucination gates → purify → train → merge → GGUF → exam → report.",
                            refresh_progress_btn: "Refresh checkpoint-based progress and ETA.",
                            student_model: "Student model for SFT and DPO training.",
                            cpu_safe: "Disable bf16 for safer CPU-only runs.",
                            saves_tag_box: "Output sub-folder under saves/. Change per experiment.",
                            do_train_check: "If unchecked, full pipeline stops after purification.",
                            gen_configs_btn: "Generate training YAML configs from current data.",
                            start_train_btn: "Start SFT and optional DPO training."
                        };

                        const applyTips = () => {
                            Object.entries(tips).forEach(([id, text]) => {
                                const el = document.getElementById(id);
                                if (el) {
                                    el.title = text;
                                    el.setAttribute("aria-label", text);
                                }
                            });
                        };

                        setTimeout(applyTips, 200);
                        setTimeout(applyTips, 1200);
                        document.addEventListener("click", applyTips);
                    })();
                    </script>
                    """
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

        # Full pipeline (all 9 stages, live streaming updates)
        run_all_btn.click(
            fn=run_full_pipeline_gen,
            inputs=[
                teachers_box, prompts_path, backend_dd, output_dir,
                max_tokens, temperature, answer_th, reason_th,
                enable_halluc, zena_gguf, profile_path_state,
                do_train_check, student_model, cpu_safe, saves_tag_box,
            ],
            outputs=[
                pipeline_log, gold_display, silver_display, drop_display,
                phase3_status, exam_html_display,
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
    import argparse

    parser = argparse.ArgumentParser(description="University of Distillation UI")
    parser.add_argument("--port", type=int, default=None, help="Server port (overrides GRADIO_SERVER_PORT)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    server_port = args.port or int(os.environ.get("GRADIO_SERVER_PORT", "7870"))
    app = build_ui()
    app.queue().launch(
        server_name="0.0.0.0",
        server_port=server_port,
        inbrowser=not args.no_browser,
        show_api=False,
    )


if __name__ == "__main__":
    main()
