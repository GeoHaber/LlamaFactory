#!/usr/bin/env python
# Copyright 2026 the LlamaFactory team.
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

"""Speed Graduation: a single-command overnight distillation pipeline.

This is the "hit one button, walk away, wake up with a trained model" path.
It chains together every stage of the Distillation Studio's Speed-Run mode
(skip teacher generation, import a pre-distilled HF dataset, train, merge,
report) so you can kick off a run before bed and come back to a finished
student in the morning.

What it does
------------
    1.  Verify / download the student base model (huggingface_hub).
    2.  Import a pre-distilled reasoning dataset from Hugging Face as a
        pre-purified GOLD set  (scripts/import_reasoning_dataset.py).
    3.  Auto-register the dataset in data/dataset_info.json so
        llamafactory-cli can find it.
    4.  Generate SFT + merge YAML configs tuned for the chosen student and
        hardware profile  (scripts/gen_distill_configs.py).
    5.  Run SFT with llamafactory-cli train.
    6.  Merge the LoRA adapter into the base model with llamafactory-cli
        export.
    7.  Write a graduation_report.md summarizing timings, artifacts, and
        how to invoke the new model.

Every stage is resumable: re-running with the same ``--tag`` skips stages
whose artifacts already exist. Stop/restart overnight runs without losing
progress.

CPU-only friendly
-----------------
When ``--cpu-safe`` is set (the default on machines without CUDA/ROCm), we:

    - disable bf16/fp16
    - shrink max_samples to keep the run within a single-night budget
    - lower num_train_epochs to 1.0
    - bump warmup_ratio slightly for stability on noisy CPU gradients

Usage
-----
    # Default overnight run -- ~2000 samples of Opus 4.6 reasoning traces
    # into Qwen2.5-1.5B-Instruct, CPU-safe, 1 epoch.
    python scripts/speed_graduation.py --tag overnight_grad

    # Smaller / faster sanity check (0.5B student, 300 samples)
    python scripts/speed_graduation.py \
        --tag sanity_grad \
        --student Qwen/Qwen2.5-0.5B-Instruct \
        --dataset TeichAI/claude-4.5-opus-high-reasoning-250x \
        --max-rows 250

    # Ambitious run on a 3B base, larger dataset
    python scripts/speed_graduation.py \
        --tag big_grad \
        --student Qwen/Qwen2.5-3B-Instruct \
        --dataset Roman1111111/claude-opus-4.6-10000x \
        --max-rows 5000 \
        --epochs 2

    # Dry-run (print the plan, don't execute)
    python scripts/speed_graduation.py --dry-run

On completion the report lives at:

    saves/<tag>/graduation_report.md
    saves/<tag>/merged/                 <- fully merged HF model, ready to load
    examples/distillation/auto/<tag>_sft.yaml
    examples/distillation/auto/<tag>_merge.yaml
    data/upstream_<tag>/consensus_sft.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths and defaults
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
DATA_DIR = ROOT / "data"
SAVES_DIR = ROOT / "saves"
CONFIG_OUT = ROOT / "examples" / "distillation" / "auto"
PYTHON = sys.executable


# Known upstream datasets we can boot from. The Studio's Speed-Run dropdown
# is the source of truth for these; keep in sync with
# scripts/import_reasoning_dataset.py::KNOWN.
KNOWN_DATASETS: dict[str, dict] = {
    "Roman1111111/claude-opus-4.6-10000x": {
        "label": "Opus-4.6 messages (~9.6k, LARGEST)",
        "approx_rows": 9600,
    },
    "nohurry/Opus-4.6-Reasoning-3000x-filtered": {
        "label": "Opus-4.6 reasoning filtered (~2.3k, SAFEST)",
        "approx_rows": 2300,
    },
    "TeichAI/claude-4.5-opus-high-reasoning-250x": {
        "label": "Opus-4.5 high-reasoning (~250, FASTEST)",
        "approx_rows": 250,
    },
    "Jackrong/Qwen3.5-reasoning-700x": {
        "label": "Qwen-rephrased reasoning (~700)",
        "approx_rows": 700,
    },
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _banner(label: str, char: str = "=") -> None:
    bar = char * 72
    print(f"\n{bar}", flush=True)
    print(f"[graduation] {label}", flush=True)
    print(f"{bar}", flush=True)


def _run_stage(label: str, cmd: list[str], log_path: Path | None = None) -> tuple[int, float]:
    """Run a stage command, streaming stdout to console and optional log file.

    Returns (exit_code, elapsed_seconds).
    """
    _banner(label)
    print(f"[graduation] $ {' '.join(str(c) for c in cmd)}", flush=True)
    t0 = time.time()

    fh = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # mode "w" so each re-run starts fresh
        fh = open(log_path, "w", encoding="utf-8")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(ROOT),
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            print(line, flush=True)
            if fh:
                fh.write(line + "\n")
                fh.flush()
        rc = proc.wait()
    finally:
        if fh:
            fh.close()

    dt = time.time() - t0
    print(f"[graduation] << {label} finished in {_fmt_duration(dt)} (exit {rc})", flush=True)
    return rc, dt


def _ensure_student_cached(repo_id: str) -> Path:
    """Make sure the student checkpoint exists on disk; download if missing.

    Uses ``snapshot_download`` with a whitelist that skips .bin / .gguf /
    other alternative formats to keep the local copy lean.
    """
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
    except ImportError as e:
        raise SystemExit(
            "huggingface_hub is required: pip install huggingface_hub"
        ) from e

    _banner(f"stage 0: ensure student model {repo_id}", "-")
    t0 = time.time()
    path = snapshot_download(
        repo_id=repo_id,
        allow_patterns=[
            "*.json",
            "*.txt",
            "*.model",
            "*.safetensors",
            "tokenizer*",
            "special_tokens_map*",
            "generation_config*",
            "merges*",
            "vocab*",
        ],
        ignore_patterns=[
            "*.bin", "*.pt", "*.pth", "*.ckpt",
            "*.gguf", "*.msgpack", "*.h5", "*.onnx",
        ],
    )
    p = Path(path)
    total_mb = sum(
        f.stat().st_size for f in p.iterdir() if f.is_file()
    ) / 1024 / 1024
    print(
        f"[graduation] student ready at {p}  "
        f"({total_mb:.0f} MB, {_fmt_duration(time.time()-t0)})",
        flush=True,
    )
    return p


def _register_dataset(dataset_name: str, jsonl_path: Path) -> None:
    """Idempotently add the new SFT dataset to data/dataset_info.json."""
    info_path = DATA_DIR / "dataset_info.json"
    try:
        info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    except (json.JSONDecodeError, ValueError):
        info = {}

    # Use a path relative to data/ if possible (LlamaFactory's dataset_dir).
    try:
        rel = jsonl_path.resolve().relative_to(DATA_DIR.resolve())
        file_name = str(rel).replace("\\", "/")
    except ValueError:
        file_name = str(jsonl_path).replace("\\", "/")

    entry = {
        "file_name": file_name,
        "columns": {"prompt": "instruction", "response": "output"},
    }

    prev = info.get(dataset_name)
    if prev == entry:
        print(f"[graduation] dataset_info.json: {dataset_name} already registered.", flush=True)
        return

    info[dataset_name] = entry
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[graduation] dataset_info.json: wrote entry for {dataset_name} -> {file_name}", flush=True)


def _patch_cpu_safe_config(yaml_path: Path, epochs: float, max_samples: int) -> None:
    """Trim the auto-generated SFT YAML for an overnight CPU run.

    Rather than fork gen_distill_configs.py we apply a tiny post-edit to the
    YAML so the upstream config stays authoritative. Idempotent.
    """
    try:
        import yaml  # noqa: PLC0415
    except ImportError as e:
        raise SystemExit("pyyaml is required: pip install pyyaml") from e

    if not yaml_path.is_file():
        print(f"[graduation] !! cannot patch missing config {yaml_path}", flush=True)
        return

    text = yaml_path.read_text(encoding="utf-8")
    # Preserve leading "### header\n\n" if any
    header = ""
    body = text
    if text.startswith("###"):
        idx = text.find("\n\n")
        if idx != -1:
            header = text[: idx + 2]
            body = text[idx + 2 :]
    cfg = yaml.safe_load(body) or {}

    cfg["num_train_epochs"] = float(epochs)
    cfg["max_samples"] = int(max_samples)
    cfg["bf16"] = False
    cfg["fp16"] = False
    # Safer on CPU: smaller cutoff keeps per-step RAM + walltime bounded.
    cfg.setdefault("cutoff_len", 2048)
    if cfg["cutoff_len"] > 4096:
        cfg["cutoff_len"] = 4096
    # No bnb 8-bit AdamW without a CUDA wheel -- fall back to plain adamw_torch.
    cfg["optim"] = "adamw_torch"
    # Save less often to save I/O time on spinning disks.
    cfg["save_steps"] = max(int(cfg.get("save_steps", 100)), 200)

    with yaml_path.open("w", encoding="utf-8") as f:
        if header:
            f.write(header)
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(
        f"[graduation] patched {yaml_path.name}: "
        f"epochs={cfg['num_train_epochs']}, "
        f"max_samples={cfg['max_samples']}, "
        f"cutoff={cfg['cutoff_len']}, "
        f"optim={cfg['optim']}, bf16=False",
        flush=True,
    )


def _rel_to_root(p: str | Path | None) -> str:
    """Return a path that's relative to ROOT when possible, else unchanged.

    This keeps the generated graduation_report.md portable instead of baking
    in the author's home directory.
    """
    if p is None:
        return ""
    try:
        return str(Path(p).resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except (ValueError, OSError):
        return str(p).replace("\\", "/")


def _write_report(report_path: Path, stats: dict) -> None:
    """Produce a human-friendly graduation_report.md summarizing the run."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    duration_total = stats.get("duration_total", 0.0)
    stages = stats.get("stages", [])
    student = stats.get("student", "?")
    dataset = stats.get("dataset", "?")
    tag = stats.get("tag", "?")
    gold = stats.get("gold_samples", 0)
    merged_dir = stats.get("merged_dir")
    sft_dir = stats.get("sft_dir")
    merged_rel = _rel_to_root(merged_dir) or f"saves/{tag}/merged"
    sft_rel = _rel_to_root(sft_dir) or f"saves/{tag}/lora/sft"

    lines: list[str] = []
    lines.append(f"# Graduation Report: {tag}")
    lines.append("")
    lines.append(f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Student base**: `{student}`")
    lines.append(f"- **Upstream dataset**: `{dataset}`")
    lines.append(f"- **GOLD samples trained**: **{gold}**")
    lines.append(f"- **Total wall-clock**: **{_fmt_duration(duration_total)}**")
    lines.append(f"- **Tag**: `{tag}`")
    lines.append("")
    lines.append("## Stage timings")
    lines.append("")
    lines.append("| Stage | Status | Duration |")
    lines.append("|:--|:--|--:|")
    for s in stages:
        name = s.get("name", "?")
        status = s.get("status", "?")
        dur = _fmt_duration(s.get("duration", 0.0))
        lines.append(f"| {name} | {status} | {dur} |")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- **LoRA adapter (SFT)**: `{sft_rel}`")
    if merged_dir:
        lines.append(f"- **Merged HF model**: `{merged_rel}`")
    lines.append(f"- **Auto SFT YAML**: `examples/distillation/auto/{tag}_sft.yaml`")
    lines.append(f"- **Auto merge YAML**: `examples/distillation/auto/{tag}_merge.yaml`")
    lines.append(f"- **Upstream GOLD jsonl**: `data/upstream_{tag}/consensus_sft.jsonl`")
    lines.append("")
    lines.append("## Next steps")
    lines.append("")
    lines.append("Load the merged model with llama.cpp / transformers:")
    lines.append("")
    lines.append("```python")
    lines.append("from transformers import AutoModelForCausalLM, AutoTokenizer")
    lines.append(f"m = AutoModelForCausalLM.from_pretrained('{merged_rel}')")
    lines.append(f"tok = AutoTokenizer.from_pretrained('{merged_rel}')")
    lines.append("```")
    lines.append("")
    lines.append("Quantize to GGUF with the Studio's GGUF stage, or directly:")
    lines.append("")
    lines.append("```bash")
    lines.append(f"python scripts/slim_down.py --model-dir saves/{tag}/merged --out-dir saves/{tag}/gguf --tag {tag} --quant q4_k_m")
    lines.append("```")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[graduation] report written -> {report_path}", flush=True)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Overnight Speed Graduation pipeline for the Distillation Studio.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
known datasets:
  Roman1111111/claude-opus-4.6-10000x         (~9.6k, LARGEST)
  nohurry/Opus-4.6-Reasoning-3000x-filtered   (~2.3k, SAFEST)
  TeichAI/claude-4.5-opus-high-reasoning-250x (~250, FASTEST)
  Jackrong/Qwen3.5-reasoning-700x             (~700)

example:
  %(prog)s --tag overnight_grad

  %(prog)s --tag sanity_grad \\
           --student Qwen/Qwen2.5-0.5B-Instruct \\
           --dataset TeichAI/claude-4.5-opus-high-reasoning-250x \\
           --max-rows 250
""",
    )
    parser.add_argument("--tag", default="overnight_grad",
                        help="Run tag -- used for saves/, configs, and dataset name.")
    parser.add_argument("--student", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Student base model (HF id).")
    parser.add_argument("--dataset", default="Roman1111111/claude-opus-4.6-10000x",
                        help="Upstream reasoning dataset (HF id).")
    parser.add_argument("--max-rows", type=int, default=2000,
                        help="Cap imported GOLD samples (overnight-safe default 2000).")
    parser.add_argument("--epochs", type=float, default=1.0,
                        help="Training epochs (default 1.0 for CPU overnight).")
    parser.add_argument("--cpu-safe", action="store_true", default=None,
                        help="Force CPU-safe overrides (no bf16, smaller cutoff).")
    parser.add_argument("--no-cpu-safe", dest="cpu_safe", action="store_false",
                        help="Disable CPU-safe overrides (GPU run).")
    parser.add_argument("--skip-dl", action="store_true",
                        help="Skip the student download check (assume it's cached).")
    parser.add_argument("--skip-merge", action="store_true",
                        help="Skip the merge stage (leave LoRA adapter unmerged).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan without executing any stage.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run through download + import + config gen, then STOP before training. "
                             "Lets you validate plumbing without a multi-hour commitment.")
    args = parser.parse_args()

    # Auto-detect CPU-safe unless overridden
    if args.cpu_safe is None:
        try:
            import torch  # noqa: PLC0415
            args.cpu_safe = not torch.cuda.is_available()
        except ImportError:
            args.cpu_safe = True

    if args.dataset in KNOWN_DATASETS:
        dataset_label = KNOWN_DATASETS[args.dataset]["label"]
    else:
        dataset_label = "(custom)"

    tag = args.tag
    upstream_dir = DATA_DIR / f"upstream_{tag}"
    sft_dataset_name = f"{tag}_consensus_sft"
    sft_yaml = CONFIG_OUT / f"{tag}_sft.yaml"
    merge_yaml = CONFIG_OUT / f"{tag}_merge.yaml"
    saves_tag_dir = SAVES_DIR / tag
    sft_output_dir = saves_tag_dir / "lora" / "sft"
    merged_output_dir = saves_tag_dir / "merged"
    report_path = saves_tag_dir / "graduation_report.md"
    logs_dir = saves_tag_dir / "speed_graduation_logs"

    _banner("Speed Graduation: the overnight one-click path")
    print(f"[graduation] tag:             {tag}", flush=True)
    print(f"[graduation] student:         {args.student}", flush=True)
    print(f"[graduation] dataset:         {args.dataset}  {dataset_label}", flush=True)
    print(f"[graduation] max-rows:        {args.max_rows}", flush=True)
    print(f"[graduation] epochs:          {args.epochs}", flush=True)
    print(f"[graduation] cpu-safe:        {args.cpu_safe}", flush=True)
    print(f"[graduation] upstream dir:    {upstream_dir}", flush=True)
    print(f"[graduation] saves dir:       {saves_tag_dir}", flush=True)
    print(f"[graduation] report target:   {report_path}", flush=True)
    if args.dry_run:
        print("[graduation] (dry-run) -- not executing any stage", flush=True)
        return 0

    stats: dict = {
        "tag": tag,
        "student": args.student,
        "dataset": args.dataset,
        "stages": [],
        "duration_total": 0.0,
        "sft_dir": str(sft_output_dir),
        "merged_dir": str(merged_output_dir) if not args.skip_merge else None,
    }
    t_start = time.time()

    # --- Stage 0: student download --------------------------------------
    if args.skip_dl:
        print("[graduation] (--skip-dl) -- trusting cache", flush=True)
        stats["stages"].append({"name": "download_student", "status": "skip", "duration": 0.0})
    else:
        t0 = time.time()
        try:
            _ensure_student_cached(args.student)
            stats["stages"].append({
                "name": "download_student",
                "status": "done",
                "duration": time.time() - t0,
            })
        except Exception as exc:  # noqa: BLE001
            print(f"[graduation] !! student download failed: {exc}", file=sys.stderr, flush=True)
            stats["stages"].append({
                "name": "download_student",
                "status": "fail",
                "duration": time.time() - t0,
            })
            return 2

    # --- Stage 1: import upstream dataset -------------------------------
    consensus_path = upstream_dir / "consensus_sft.jsonl"
    if consensus_path.is_file() and consensus_path.stat().st_size > 0:
        n_rows = sum(1 for _ in consensus_path.read_text(encoding="utf-8").splitlines() if _.strip())
        print(f"[graduation] (resume) consensus_sft.jsonl exists with {n_rows} rows -- reusing", flush=True)
        stats["gold_samples"] = n_rows
        stats["stages"].append({"name": "import_dataset", "status": "skip", "duration": 0.0})
    else:
        cmd_import = [
            PYTHON, str(SCRIPTS / "import_reasoning_dataset.py"),
            "--dataset", args.dataset,
            "--output-dir", str(upstream_dir),
            "--tier", "GOLD",
            "--max-rows", str(int(args.max_rows)),
        ]
        rc, dt = _run_stage("stage 1: import upstream dataset", cmd_import,
                            log_path=logs_dir / "01_import.log")
        stats["stages"].append({
            "name": "import_dataset",
            "status": "done" if rc == 0 else "fail",
            "duration": dt,
        })
        if rc != 0:
            return 3
        try:
            rpt = json.loads((upstream_dir / "purification_report.json").read_text(encoding="utf-8"))
            stats["gold_samples"] = int(rpt.get("gold_count", 0))
        except (OSError, json.JSONDecodeError, ValueError):
            stats["gold_samples"] = 0

    if stats.get("gold_samples", 0) == 0:
        print("[graduation] !! GOLD count is 0 -- nothing to train. Aborting.", file=sys.stderr, flush=True)
        return 4

    # --- Stage 2: register dataset in data/dataset_info.json ------------
    _banner("stage 2: register dataset", "-")
    t0 = time.time()
    _register_dataset(sft_dataset_name, consensus_path)
    stats["stages"].append({
        "name": "register_dataset",
        "status": "done",
        "duration": time.time() - t0,
    })

    # --- Stage 3: generate auto configs ---------------------------------
    if sft_yaml.is_file():
        print(f"[graduation] (resume) {sft_yaml.name} exists -- reusing", flush=True)
        stats["stages"].append({"name": "gen_configs", "status": "skip", "duration": 0.0})
    else:
        cmd_cfg = [
            PYTHON, str(SCRIPTS / "gen_distill_configs.py"),
            "--student", args.student,
            "--data-dir", str(upstream_dir),
            "--out-dir", str(CONFIG_OUT),
            "--tag", tag,
            "--sft-dataset-name", sft_dataset_name,
        ]
        if args.cpu_safe:
            cmd_cfg.append("--cpu-safe")
        rc, dt = _run_stage("stage 3: generate auto configs", cmd_cfg,
                            log_path=logs_dir / "03_gen_configs.log")
        stats["stages"].append({
            "name": "gen_configs",
            "status": "done" if rc == 0 else "fail",
            "duration": dt,
        })
        if rc != 0:
            return 5
        if not sft_yaml.is_file():
            print(f"[graduation] !! config generation succeeded but {sft_yaml} missing.", file=sys.stderr, flush=True)
            return 6

    # Post-patch the config for overnight CPU constraints.
    _banner("stage 3b: patch config for overnight CPU profile", "-")
    t0 = time.time()
    _patch_cpu_safe_config(sft_yaml, epochs=args.epochs, max_samples=int(args.max_rows))
    stats["stages"].append({
        "name": "patch_config",
        "status": "done",
        "duration": time.time() - t0,
    })

    if args.smoke_test:
        stats["duration_total"] = time.time() - t_start
        _banner(f"SMOKE-TEST OK: stopped before training. ({_fmt_duration(stats['duration_total'])})")
        print(f"[graduation] sft yaml:   {sft_yaml}", flush=True)
        print(f"[graduation] merge yaml: {merge_yaml}", flush=True)
        print(f"[graduation] gold jsonl: {consensus_path}", flush=True)
        print("[graduation] To actually train, re-run without --smoke-test.", flush=True)
        # Write a smoke-test report so we know plumbing worked
        _write_report(report_path, stats)
        return 0

    # --- Stage 4: SFT train ---------------------------------------------
    sft_done_marker = sft_output_dir / "adapter_config.json"
    if sft_done_marker.is_file():
        print(f"[graduation] (resume) SFT adapter exists at {sft_output_dir} -- skipping train", flush=True)
        stats["stages"].append({"name": "sft_train", "status": "skip", "duration": 0.0})
    else:
        cmd_train = [
            PYTHON, "-m", "llamafactory.cli", "train", str(sft_yaml),
        ]
        env_note = ("This is the heavy stage. On CPU expect many hours; "
                    "on a GPU it's minutes. Safe to Ctrl-C and resume -- "
                    "LlamaFactory writes a checkpoint every save_steps.")
        print(f"[graduation] note: {env_note}", flush=True)
        rc, dt = _run_stage("stage 4: SFT train", cmd_train,
                            log_path=logs_dir / "04_sft_train.log")
        stats["stages"].append({
            "name": "sft_train",
            "status": "done" if rc == 0 else "fail",
            "duration": dt,
        })
        if rc != 0:
            # Write a partial report so you know where it died.
            stats["duration_total"] = time.time() - t_start
            _write_report(report_path, stats)
            return 7

    # --- Stage 5: merge (optional) --------------------------------------
    if args.skip_merge:
        print("[graduation] (--skip-merge) -- leaving LoRA adapter unmerged", flush=True)
        stats["stages"].append({"name": "merge", "status": "skip", "duration": 0.0})
    else:
        merged_marker = merged_output_dir / "config.json"
        if merged_marker.is_file():
            print(f"[graduation] (resume) merged model exists at {merged_output_dir} -- skipping merge", flush=True)
            stats["stages"].append({"name": "merge", "status": "skip", "duration": 0.0})
        elif not merge_yaml.is_file():
            print(f"[graduation] !! merge yaml missing: {merge_yaml}", file=sys.stderr, flush=True)
            stats["stages"].append({"name": "merge", "status": "fail", "duration": 0.0})
        else:
            cmd_merge = [
                PYTHON, "-m", "llamafactory.cli", "export", str(merge_yaml),
            ]
            rc, dt = _run_stage("stage 5: merge LoRA into base", cmd_merge,
                                log_path=logs_dir / "05_merge.log")
            stats["stages"].append({
                "name": "merge",
                "status": "done" if rc == 0 else "fail",
                "duration": dt,
            })
            if rc != 0:
                stats["duration_total"] = time.time() - t_start
                _write_report(report_path, stats)
                return 8

    # --- Stage 6: write graduation report --------------------------------
    stats["duration_total"] = time.time() - t_start
    _write_report(report_path, stats)

    _banner(f"DONE: Speed Graduation '{tag}' in {_fmt_duration(stats['duration_total'])}")
    print(f"[graduation] merged model: {merged_output_dir}", flush=True)
    print(f"[graduation] report:       {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
