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

"""ZEN Forge — Multi-Teacher Distillation Wizard.

A clean 3-phase wizard UI:
  Phase 1  Setup   → pick teachers, prompts, backend → "Start Teaching"
  Phase 2  Purify  → review GOLD/SILVER/DROP → "Accept & Generate Configs"
  Phase 3  Train   → generated YAML paths, student picker → "Start Training"

Launch:  python scripts/distill_ui.py
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import gradio as gr  # xray: ignore[SEC-015]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parent
PYTHON = sys.executable

# ---------------------------------------------------------------------------
# CSS — minimal dark/light theme, Zena branding
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
:root {
    --zen-accent: #7c3aed;
    --zen-accent-hover: #6d28d9;
    --zen-success: #16a34a;
    --zen-warn: #d97706;
    --zen-danger: #dc2626;
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
.zen-phase-title {
    font-size: 1.15rem; font-weight: 600; margin: 8px 0 4px 0;
    color: var(--zen-accent);
}
.zen-stat-card {
    padding: 16px; border-radius: 10px; text-align: center;
    border: 1px solid var(--border-color-primary, #e5e7eb);
}
.zen-stat-card h3 { margin: 0 0 4px 0; font-size: 2rem; }
.zen-stat-card p  { margin: 0; opacity: 0.7; font-size: 0.85rem; }
.zen-gold   h3 { color: #ca8a04; }
.zen-silver h3 { color: #64748b; }
.zen-drop   h3 { color: var(--zen-danger); }
.zen-hidden { display: none !important; }
"""

# ---------------------------------------------------------------------------
# i18n — just en/zh for now (easy to extend)
# ---------------------------------------------------------------------------
LANG = {
    "en": {
        "title": "ZEN Forge — Distillation Wizard",
        "phase1": "Phase 1 · Setup Teachers",
        "phase2": "Phase 2 · Purify Knowledge",
        "phase3": "Phase 3 · Train Student",
        "teachers_label": "Teacher GGUF paths (one per line)",
        "teachers_info": "Local .gguf files. Use 1-5 teachers (odd count = majority vote).",
        "prompts_label": "Prompts file",
        "prompts_info": "JSONL with {\"id\": ..., \"prompt\": ...} rows.",
        "backend_label": "Backend",
        "backend_info": "inprocess = direct GGUF (zen_core_libs). server = HTTP llama-server.",
        "output_label": "Output directory",
        "start_teach": "Start Teaching",
        "accept_purify": "Accept & Generate Configs",
        "start_train": "Start Training",
        "settings": "Settings",
        "answer_th": "Answer consensus threshold",
        "reason_th": "Reasoning alignment threshold",
        "max_tokens": "Max tokens per response",
        "temperature": "Sampling temperature",
        "student_label": "Student model (HF name or local path)",
        "cpu_safe": "CPU-safe mode (disable bf16)",
        "gold": "GOLD (SFT)",
        "silver": "SILVER (DPO)",
        "drop": "DROP",
        "done": "Done",
        "running": "Running…",
        "waiting": "Waiting for Phase 1…",
        "waiting3": "Waiting for Phase 2…",
    },
    "zh": {
        "title": "ZEN Forge — 蒸馏向导",
        "phase1": "阶段 1 · 配置教师",
        "phase2": "阶段 2 · 知识净化",
        "phase3": "阶段 3 · 训练学生",
        "teachers_label": "教师 GGUF 路径（每行一个）",
        "teachers_info": "本地 .gguf 文件。使用 1-5 个教师（奇数 = 多数投票）。",
        "prompts_label": "提示词文件",
        "prompts_info": "JSONL 文件，每行 {\"id\": ..., \"prompt\": ...}。",
        "backend_label": "后端",
        "backend_info": "inprocess = 直接 GGUF。server = HTTP llama-server。",
        "output_label": "输出目录",
        "start_teach": "开始教学",
        "accept_purify": "接受并生成配置",
        "start_train": "开始训练",
        "settings": "设置",
        "answer_th": "答案共识阈值",
        "reason_th": "推理对齐阈值",
        "max_tokens": "每次响应最大 token",
        "temperature": "采样温度",
        "student_label": "学生模型（HF 名称或本地路径）",
        "cpu_safe": "CPU 安全模式（禁用 bf16）",
        "gold": "GOLD（SFT）",
        "silver": "SILVER（DPO）",
        "drop": "DROP",
        "done": "完成",
        "running": "运行中…",
        "waiting": "等待阶段 1…",
        "waiting3": "等待阶段 2…",
    },
}


def t(key: str, lang: str = "en") -> str:
    return LANG.get(lang, LANG["en"]).get(key, key)


# ---------------------------------------------------------------------------
# Backend helpers — call the existing scripts as subprocesses
# ---------------------------------------------------------------------------

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


def _profile_teachers(
    teachers_text: str, backend: str, output_dir: str, do_probe: bool,
) -> tuple[str, str]:
    """Run teacher_profiler.py and return (markdown_table, profile_json_path)."""
    lines = [l.strip() for l in teachers_text.strip().splitlines() if l.strip()]
    if not lines:
        return "**Error:** no teacher paths provided.", ""

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
        err = stderr[-2000:] if stderr else "unknown error"
        return f"**Profiling failed** (exit {rc})\n```\n{err}\n```", ""

    # Build a readable capability matrix table
    try:
        with open(profile_path, encoding="utf-8") as f:
            profile = json.load(f)
        matrix = profile.get("capability_matrix", {})
        if not matrix:
            return "**Profiling done** (no capability data).", profile_path

        # Collect all capabilities
        all_caps = sorted({c for scores in matrix.values() for c in scores})
        # Header
        hdr = "| Teacher | " + " | ".join(c.title() for c in all_caps) + " |"
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
                    cells.append(f"~{v:.0%}~")
            rows.append(f"| {short} | " + " | ".join(cells) + " |")

        md = "### Teacher Expertise Matrix\n\n" + "\n".join([hdr, sep] + rows)
        md += "\n\n*✓ = strong (≥70%), ~strikethrough~ = weak (<30%). Prompts will be routed to qualified teachers.*"
        return md, profile_path

    except Exception as e:  # xray: ignore[QUAL-011]
        return f"**Profiling done** but couldn't parse results: {e}", profile_path


def _run_script(cmd: list[str], timeout: int = 7200) -> tuple[int, str, str]:
    """Run a script and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, cwd=str(ROOT_DIR)
    )
    return result.returncode, result.stdout, result.stderr


# ---------------------------------------------------------------------------
# Phase 1 — Generate teacher responses
# ---------------------------------------------------------------------------

def run_phase1(
    teachers_text: str,
    prompts_path: str,
    backend: str,
    output_dir: str,
    max_tokens: int,
    temperature: float,
    profile_path: str,
    lang: str,
):
    """Execute generation with optional smart routing, return (status_md, phase2_visible, phase2_info)."""
    if not teachers_text.strip():
        return "**Error:** no teacher models provided.", gr.update(visible=False), ""
    if not prompts_path.strip() or not Path(prompts_path).is_file():
        return f"**Error:** prompts file not found: `{prompts_path}`", gr.update(visible=False), ""

    try:
        manifest_path = _write_manifest(teachers_text, backend, output_dir)
    except ValueError as e:
        return f"**Error:** {e}", gr.update(visible=False), ""

    out_jsonl = str(Path(output_dir) / "teacher_responses.jsonl")
    cmd = [
        PYTHON, str(SCRIPTS_DIR / "multi_teacher_generate.py"),
        "--manifest", manifest_path,
        "--prompts", prompts_path,
        "--out", out_jsonl,
        "--max-tokens", str(int(max_tokens)),
        "--temperature", str(temperature),
    ]
    # Smart routing via capability profile
    if profile_path and Path(profile_path).is_file():
        cmd.extend(["--profile", profile_path])

    rc, stdout, stderr = _run_script(cmd)
    if rc != 0:
        err = stderr[-2000:] if stderr else "unknown error"
        return f"**Generation failed** (exit {rc})\n```\n{err}\n```", gr.update(visible=False), ""

    # Count results
    count = 0
    if Path(out_jsonl).is_file():
        with open(out_jsonl, encoding="utf-8") as f:
            count = sum(1 for l in f if l.strip())

    routing_note = " (smart routing)" if profile_path else ""
    status = f"**{t('done', lang)}** — {count} samples generated{routing_note} → `{out_jsonl}`"
    return status, gr.update(visible=True), out_jsonl


# ---------------------------------------------------------------------------
# Phase 2 — Purify
# ---------------------------------------------------------------------------

def run_phase2(
    output_dir: str,
    answer_th: float,
    reason_th: float,
    lang: str,
):
    """Execute purification, return (status_md, gold_n, silver_n, drop_n, phase3_visible, config_info)."""
    responses_file = Path(output_dir) / "teacher_responses.jsonl"
    if not responses_file.is_file():
        return "**Error:** no teacher_responses.jsonl found. Run Phase 1 first.", 0, 0, 0, gr.update(visible=False), ""

    cmd = [
        PYTHON, str(SCRIPTS_DIR / "purify_teacher_outputs.py"),
        "--input", str(responses_file),
        "--out-dir", output_dir,
        "--answer-threshold", str(answer_th),
        "--reason-threshold", str(reason_th),
    ]
    rc, stdout, stderr = _run_script(cmd, timeout=600)
    if rc != 0:
        err = stderr[-2000:] if stderr else "unknown error"
        return f"**Purification failed** (exit {rc})\n```\n{err}\n```", 0, 0, 0, gr.update(visible=False), ""

    # Read report
    report_path = Path(output_dir) / "purification_report.json"
    gold_n = silver_n = drop_n = 0
    if report_path.is_file():
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)  # xray: ignore[PY-005]
        gold_n = report.get("gold_count", 0)
        silver_n = report.get("silver_count", 0)
        drop_n = report.get("dropped_count", 0)

    status = f"**{t('done', lang)}** — Purified into {gold_n} GOLD, {silver_n} SILVER, {drop_n} DROP"
    return status, gold_n, silver_n, drop_n, gr.update(visible=True), output_dir


# ---------------------------------------------------------------------------
# Phase 2b — Generate configs
# ---------------------------------------------------------------------------

def run_gen_configs(
    output_dir: str,
    student_model: str,
    cpu_safe: bool,
    lang: str,
):
    """Generate training YAMLs and return status + file list markdown."""
    if not student_model.strip():
        return "**Error:** please specify a student model."

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
        err = stderr[-2000:] if stderr else "unknown error"
        return f"**Config generation failed**\n```\n{err}\n```"

    # List generated files
    yamls = sorted(glob.glob(os.path.join(config_out, "*.yaml")))
    lines = [f"**{t('done', lang)}** — Generated configs:\n"]
    for y in yamls:
        lines.append(f"- `{y}`")
    lines.append(f"\n**Next:** click **{t('start_train', lang)}** or run manually:")
    lines.append(f"```\nllamafactory-cli train {yamls[0] if yamls else config_out}\n```")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 3 — Train (kicks off llamafactory-cli)
# ---------------------------------------------------------------------------

def run_phase3(
    output_dir: str,
    student_model: str,
    cpu_safe: bool,
    lang: str,
):
    """Run SFT + DPO training via llamafactory-cli."""
    config_out = ROOT_DIR / "examples" / "distillation" / "auto"
    sft_yaml = config_out / "distill_auto_sft.yaml"
    dpo_yaml = config_out / "distill_auto_dpo.yaml"

    if not sft_yaml.is_file():
        return "**Error:** SFT config not found. Run 'Accept & Generate Configs' first."

    lines = [f"**{t('running', lang)}** SFT training…\n"]

    # SFT
    cmd_sft = ["llamafactory-cli", "train", str(sft_yaml)]
    rc, stdout, stderr = _run_script(cmd_sft, timeout=14400)
    if rc != 0:
        return f"**SFT failed** (exit {rc})\n```\n{stderr[-2000:]}\n```"
    lines.append(f"SFT **{t('done', lang)}**\n")

    # DPO (only if config exists and has data)
    if dpo_yaml.is_file():
        lines.append(f"**{t('running', lang)}** DPO training…\n")
        cmd_dpo = ["llamafactory-cli", "train", str(dpo_yaml)]
        rc, stdout, stderr = _run_script(cmd_dpo, timeout=14400)
        if rc != 0:
            lines.append(f"DPO **failed** (exit {rc})\n```\n{stderr[-1500:]}\n```")
        else:
            lines.append(f"DPO **{t('done', lang)}**\n")

    lines.append("\n**Pipeline complete.** Check `saves/distill_auto/` for outputs.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build the UI
# ---------------------------------------------------------------------------

def build_ui(lang: str = "en") -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.violet,
        secondary_hue=gr.themes.colors.gray,
    )

    with gr.Blocks(
        title=t("title", lang),
        theme=theme,
        css=CUSTOM_CSS,
    ) as app:

        # ── Header ──────────────────────────────────────────────
        gr.HTML(
            '<div class="zen-header">'
            '  <span class="zen-logo">⚡ ZEN Forge</span>'
            f'  <span style="opacity:0.6; font-size:0.85rem;">{t("title", lang)}</span>'
            "</div>"
        )

        # ── Settings (collapsible) ──────────────────────────────
        with gr.Accordion(t("settings", lang), open=False):
            with gr.Row():
                lang_dd = gr.Dropdown(
                    choices=["en", "zh"], value=lang, label="Language",
                    info="Interface language", scale=1,
                )
                max_tokens = gr.Slider(
                    64, 4096, value=512, step=64,
                    label=t("max_tokens", lang),
                    info="Controls max length of each teacher response.",
                    scale=2,
                )
                temperature = gr.Slider(
                    0.0, 2.0, value=0.7, step=0.05,
                    label=t("temperature", lang),
                    info="Lower = more deterministic. Higher = more creative.",
                    scale=2,
                )
            with gr.Row():
                answer_th = gr.Slider(
                    0.5, 1.0, value=0.85, step=0.01,
                    label=t("answer_th", lang),
                    info="N-gram similarity needed for teachers to 'agree'. Higher = stricter GOLD gate.",
                )
                reason_th = gr.Slider(
                    0.2, 1.0, value=0.60, step=0.01,
                    label=t("reason_th", lang),
                    info="Similarity for reasoning traces to align. Below this → SILVER (DPO pair).",
                )

        # ── Phase 1 ─────────────────────────────────────────────
        gr.HTML(f'<div class="zen-phase-title">{t("phase1", lang)}</div>')

        with gr.Row():
            teachers_box = gr.Textbox(
                lines=5, scale=3,
                label=t("teachers_label", lang),
                info=t("teachers_info", lang),
                placeholder="C:/AI/Models/deepseek-r1-8b-q8_0.gguf\nC:/AI/Models/qwen2.5-3b-instruct-q8_0.gguf",  # xray: ignore[PORT-002]
            )
            with gr.Column(scale=2):
                prompts_path = gr.Textbox(
                    label=t("prompts_label", lang),
                    info=t("prompts_info", lang),
                    placeholder="data/distill_prompts.jsonl",
                )
                with gr.Row():
                    backend_dd = gr.Dropdown(
                        choices=["inprocess", "server"], value="inprocess",
                        label=t("backend_label", lang),
                        info=t("backend_info", lang),
                    )
                    output_dir = gr.Textbox(
                        value="data/purified",
                        label=t("output_label", lang),
                    )

        with gr.Row():
            profile_btn = gr.Button(
                "Profile Teachers", variant="secondary", size="lg", scale=1,
            )
            probe_check = gr.Checkbox(
                label="Run mini-benchmark probe",
                value=False,
                info="Sends 5 test prompts per capability to each teacher. Slower but more accurate.",
                scale=1,
            )
        profile_status = gr.Markdown(value="")
        profile_path_state = gr.State("")

        start_teach_btn = gr.Button(
            t("start_teach", lang), variant="primary", size="lg",
        )
        phase1_status = gr.Markdown(value="")

        # ── Phase 2 (hidden until Phase 1 done) ────────────────
        with gr.Column(visible=False) as phase2_block:
            gr.HTML(f'<div class="zen-phase-title">{t("phase2", lang)}</div>')

            with gr.Row():
                gold_display = gr.Number(label=t("gold", lang), value=0, interactive=False)
                silver_display = gr.Number(label=t("silver", lang), value=0, interactive=False)
                drop_display = gr.Number(label=t("drop", lang), value=0, interactive=False)

            phase2_status = gr.Markdown(value="")

            with gr.Row():
                student_model = gr.Textbox(
                    value="Qwen/Qwen2.5-1.5B-Instruct",
                    label=t("student_label", lang),
                    info="HuggingFace model ID or local path for the student.",
                    scale=3,
                )
                cpu_safe = gr.Checkbox(
                    value=False,
                    label=t("cpu_safe", lang),
                    info="Check if you have no GPU. Disables bf16.",
                )

            accept_btn = gr.Button(
                t("accept_purify", lang), variant="primary", size="lg",
            )
            config_status = gr.Markdown(value="")

        # ── Phase 3 (hidden until Phase 2 done) ────────────────
        with gr.Column(visible=False) as phase3_block:
            gr.HTML(f'<div class="zen-phase-title">{t("phase3", lang)}</div>')

            start_train_btn = gr.Button(
                t("start_train", lang), variant="primary", size="lg",
            )
            phase3_status = gr.Markdown(value="")

        # ── Internal state to pass output_dir between phases ───
        responses_path_state = gr.State("")
        purified_dir_state = gr.State("")

        # ── Wiring ──────────────────────────────────────────────

        def _profile_wrapper(teachers, backend, outdir, do_probe):
            md, prof_path = _profile_teachers(teachers, backend, outdir, do_probe)
            return md, prof_path

        profile_btn.click(
            fn=_profile_wrapper,
            inputs=[teachers_box, backend_dd, output_dir, probe_check],
            outputs=[profile_status, profile_path_state],
        )

        def _phase1_wrapper(teachers, prompts, backend, outdir, maxtok, temp, prof_path, lang_val):
            status, vis_update, resp_path = run_phase1(
                teachers, prompts, backend, outdir, maxtok, temp, prof_path, lang_val
            )
            # Also run purification immediately
            if resp_path:
                p2_status, g, s, d, p3_vis, pdir = run_phase2(
                    outdir, 0.85, 0.60, lang_val
                )
                return (
                    status,
                    gr.update(visible=True),  # phase2_block
                    g, s, d,
                    p2_status,
                    resp_path,
                    outdir,
                )
            return (
                status,
                gr.update(visible=False),
                0, 0, 0, "",
                "", "",
            )

        start_teach_btn.click(
            fn=_phase1_wrapper,
            inputs=[teachers_box, prompts_path, backend_dd, output_dir, max_tokens, temperature, profile_path_state, lang_dd],
            outputs=[
                phase1_status,
                phase2_block,
                gold_display, silver_display, drop_display,
                phase2_status,
                responses_path_state,
                purified_dir_state,
            ],
        )

        def _phase2_wrapper(outdir, student, cpu, lang_val):
            cfg_status = run_gen_configs(outdir, student, cpu, lang_val)
            return cfg_status, gr.update(visible=True)

        accept_btn.click(
            fn=_phase2_wrapper,
            inputs=[output_dir, student_model, cpu_safe, lang_dd],
            outputs=[config_status, phase3_block],
        )

        def _phase3_wrapper(outdir, student, cpu, lang_val):
            return run_phase3(outdir, student, cpu, lang_val)

        start_train_btn.click(
            fn=_phase3_wrapper,
            inputs=[output_dir, student_model, cpu_safe, lang_dd],
            outputs=[phase3_status],
        )

    return app


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main():
    app = build_ui("en")
    app.queue().launch(
        server_name="0.0.0.0",
        server_port=7870,
        inbrowser=True,
        show_api=False,
    )


if __name__ == "__main__":
    main()
