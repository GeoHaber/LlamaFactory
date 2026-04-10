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

r"""Rank Probe — empirical LoRA rank selection via short training sweeps.

Runs 50-step training at candidate ranks (default [16, 32, 64, 128]) on a
small sample of the skill data, measures loss convergence slope, and selects
the lowest rank whose slope is within 5% of the steepest (best) slope.

This directly replaces the static ``_SKILL_RANK_DEFAULTS`` lookup in
``skill_branch.py`` with an empirically grounded selection.

Rationale (from arXiv 2603.02224 "Subspace Geometry"):
    The optimal rank depends on the *subspace angle* between the skill's
    gradient subspace and the trunk's learned subspace.  A trunk that already
    saw some code data needs less rank for a code branch than a trunk that
    saw no code at all.  The static table can't know this — probing can.

Usage::

    # Standalone
    python scripts/rank_probe.py \
        --trunk saves/trunk_v1/merged \
        --dataset my_skill_branch \
        --tag probe_translate

    # Custom ranks and more steps
    python scripts/rank_probe.py \
        --trunk saves/trunk_v1/merged \
        --dataset my_ds \
        --tag probe_custom \
        --ranks 8,16,32,64,128,256 \
        --steps 100

    # Programmatic (from skill_branch.py --rank auto)
    from rank_probe import probe_rank
    best_rank, details = probe_rank(
        "saves/trunk_v1/merged", "my_skill_branch", "probe_tag"
    )
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml  # xray: ignore[SEC-015]

ROOT = Path(__file__).resolve().parent.parent

# ── Defaults ─────────────────────────────────────────────────────────────────
PROBE_RANKS: list[int] = [16, 32, 64, 128]
PROBE_STEPS: int = 50
WARMUP_SKIP: int = 5          # ignore first N steps (warmup noise)
SLOPE_TOLERANCE: float = 0.05  # 5% — pick lowest rank within this of best
MAX_SAMPLES: int = 500         # keep each probe fast


# ── Logging ──────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f"[rank_probe] {msg}", flush=True)  # xray: ignore[PY-004]


# ── YAML builders ────────────────────────────────────────────────────────────

def _probe_yaml(
    trunk_path: str,
    dataset_name: str,
    rank: int,
    tag: str,
    cpu_safe: bool = False,
    steps: int = PROBE_STEPS,
) -> dict:
    """Minimal LoRA SFT config for a single rank-probe run.

    Key differences from a real branch training config:

    * ``max_steps`` instead of ``num_train_epochs`` (short fixed-step run)
    * ``logging_steps=1`` (per-step loss for slope computation)
    * ``max_samples`` capped at 500 (speed; slope only needs a sample)
    * ``overwrite_output_dir=True`` (re-runnable without manual cleanup)
    * ``save_steps`` huge (no checkpoint writes — we only need the log)
    """
    return {
        "model_name_or_path": trunk_path,
        "trust_remote_code": True,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": rank,
        "lora_alpha": rank * 2,        # α = 2r convention
        "lora_dropout": 0.0,
        "lora_target": "all",
        # ── Dataset ──────────────────────────────────────────────────────
        "dataset_dir": "data",
        "dataset": dataset_name,
        "template": "qwen3_5",
        "enable_thinking": True,
        "mask_history": True,
        "cutoff_len": 2048,              # short for speed
        "max_samples": MAX_SAMPLES,
        "preprocessing_num_workers": 1,
        "dataloader_num_workers": 0,
        # ── Output ───────────────────────────────────────────────────────
        "output_dir": f"saves/_probes/{tag}_r{rank}",
        "logging_steps": 1,             # log every step
        "save_steps": 999999,           # no checkpoint saves
        "plot_loss": False,
        "overwrite_output_dir": True,
        "report_to": "none",
        # ── Optimizer ────────────────────────────────────────────────────
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1.0e-4,
        "max_steps": steps,
        "lr_scheduler_type": "cosine",
        "warmup_steps": WARMUP_SKIP,
        "optim": "adamw_torch" if cpu_safe else "adamw_8bit",
        "weight_decay": 0.001,
        "bf16": not cpu_safe,
        "fp16": False,
    }


def _probe_oft_yaml(
    trunk_path: str,
    dataset_name: str,
    oft_rank: int,
    tag: str,
    oft_block_size: int = 32,
    cpu_safe: bool = False,
    steps: int = PROBE_STEPS,
) -> dict:
    """Minimal OFT config for a rank-probe run.

    OFT (Orthogonal Fine-Tuning) constrains updates to orthogonal rotations,
    which provably preserves pre-trained representations.  The ``oft_rank``
    controls the intrinsic dimension of the rotation (NOT the same as LoRA
    rank — OFT rank is typically much smaller).
    """
    return {
        "model_name_or_path": trunk_path,
        "trust_remote_code": True,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "oft",
        "oft_rank": oft_rank,
        "oft_block_size": oft_block_size,
        "oft_target": "all",
        "module_dropout": 0.0,
        # ── Dataset ──────────────────────────────────────────────────────
        "dataset_dir": "data",
        "dataset": dataset_name,
        "template": "qwen3_5",
        "enable_thinking": True,
        "mask_history": True,
        "cutoff_len": 2048,
        "max_samples": MAX_SAMPLES,
        "preprocessing_num_workers": 1,
        "dataloader_num_workers": 0,
        # ── Output ───────────────────────────────────────────────────────
        "output_dir": f"saves/_probes/{tag}_oft_r{oft_rank}",
        "logging_steps": 1,
        "save_steps": 999999,
        "plot_loss": False,
        "overwrite_output_dir": True,
        "report_to": "none",
        # ── Optimizer ────────────────────────────────────────────────────
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1.0e-4,
        "max_steps": steps,
        "lr_scheduler_type": "cosine",
        "warmup_steps": WARMUP_SKIP,
        "optim": "adamw_torch" if cpu_safe else "adamw_8bit",
        "weight_decay": 0.001,
        "bf16": not cpu_safe,
        "fp16": False,
    }


# ── Training log reader ─────────────────────────────────────────────────────

def _read_trainer_log(output_dir: str | Path) -> list[tuple[int, float]]:
    """Read (step, loss) pairs from the training output.

    Tries two sources in order:

    1. ``trainer_log.jsonl`` — LlamaFactory's custom LogCallback output
    2. ``trainer_state.json`` — HuggingFace Trainer's state with log_history

    Returns a list of ``(step, loss)`` sorted by step.
    """
    output_dir = Path(output_dir)
    pairs: list[tuple[int, float]] = []

    # Source 1: LlamaFactory trainer_log.jsonl
    log_path = output_dir / "trainer_log.jsonl"
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = entry.get("current_steps") or entry.get("step")
            loss = entry.get("loss")
            if step is not None and loss is not None:
                pairs.append((int(step), float(loss)))

    # Source 2 (fallback): HuggingFace trainer_state.json
    if not pairs:
        state_path = output_dir / "trainer_state.json"
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                for entry in state.get("log_history", []):
                    step = entry.get("step")
                    loss = entry.get("loss")
                    if step is not None and loss is not None:
                        pairs.append((int(step), float(loss)))
            except (json.JSONDecodeError, KeyError):
                pass

    pairs.sort(key=lambda x: x[0])
    return pairs


# ── Slope computation ────────────────────────────────────────────────────────

def _linear_slope(points: list[tuple[int, float]]) -> float:
    """Ordinary least-squares slope of ``(x, y)`` points.

    Returns slope.  Negative → loss decreasing → good convergence.
    """
    n = len(points)
    if n < 3:
        return 0.0

    x_mean = sum(x for x, _ in points) / n
    y_mean = sum(y for _, y in points) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in points)
    den = sum((x - x_mean) ** 2 for x, _ in points)
    if abs(den) < 1e-12:
        return 0.0
    return num / den


def _loss_slope(losses: list[tuple[int, float]], warmup: int = WARMUP_SKIP) -> float:
    """Compute loss convergence slope, skipping warmup steps.

    Args:
        losses: ``(step, loss)`` pairs from training.
        warmup: Discard steps ≤ this value (warmup noise distorts the slope).

    Returns:
        Slope (negative = loss is decreasing = learning is happening).
    """
    post_warmup = [(s, l) for s, l in losses if s > warmup]
    return _linear_slope(post_warmup)


# ── Subprocess runner ────────────────────────────────────────────────────────

def _run_probe(yaml_path: Path, log_path: Path) -> int:
    """Run one probe training and return exit code."""
    cmd = [sys.executable, "-m", "llamafactory.cli", "train", str(yaml_path)]
    _log(f"  $ {' '.join(cmd[:4])} ...")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(ROOT),
        )
        return proc.wait()


# ── Core probe function (importable) ─────────────────────────────────────────

def probe_rank(
    trunk_path: str,
    dataset_name: str,
    tag: str,
    cpu_safe: bool = False,
    ranks: list[int] | None = None,
    steps: int = PROBE_STEPS,
    tolerance: float = SLOPE_TOLERANCE,
    cleanup: bool = True,
) -> tuple[int, dict[int, dict]]:
    """Run short training at each candidate rank and return the optimal one.

    Selection rule: pick the **lowest** rank whose loss-convergence slope is
    within ``tolerance`` (default 5%) of the steepest (most negative) slope
    observed.  This finds the smallest adapter that still learns effectively.

    Args:
        trunk_path:   HF model dir (merged trunk).
        dataset_name: Name registered in ``data/dataset_info.json``.
        tag:          Unique tag for this probe session.
        cpu_safe:     CPU-friendly config (no bf16, adamw_torch).
        ranks:        Candidate ranks to test (default ``[16, 32, 64, 128]``).
        steps:        Training steps per probe (default 50).
        tolerance:    Slope tolerance for selection (default 0.05 = 5%).
        cleanup:      Remove probe output dirs after finishing.

    Returns:
        ``(best_rank, details)`` where *details* maps
        ``rank → {slope, final_loss, n_points, time_s, ok}``.
    """
    if ranks is None:
        ranks = list(PROBE_RANKS)

    _log(f"probing ranks {ranks} on dataset '{dataset_name}' ({steps} steps each)")
    results: dict[int, dict] = {}
    probe_dirs: list[Path] = []

    for rank in ranks:
        _log(f"-- rank {rank} --")
        t0 = time.time()

        # Build config & paths
        cfg = _probe_yaml(trunk_path, dataset_name, rank, tag, cpu_safe, steps)
        out_dir = ROOT / cfg["output_dir"]
        probe_dirs.append(out_dir)

        yaml_path = (
            ROOT / "examples" / "distillation" / "auto" / f"_probe_{tag}_r{rank}.yaml"
        )
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with yaml_path.open("w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        # Train
        log_path = ROOT / "saves" / "_probes" / "logs" / f"{tag}_r{rank}.log"
        rc = _run_probe(yaml_path, log_path)
        elapsed = time.time() - t0

        if rc != 0:
            _log(f"  FAILED (rc={rc}) — skipping.  See {log_path}")
            results[rank] = {
                "slope": 0.0, "final_loss": float("inf"),
                "n_points": 0, "time_s": elapsed, "ok": False,
            }
            yaml_path.unlink(missing_ok=True)
            continue

        # Read & score
        losses = _read_trainer_log(out_dir)
        slope = _loss_slope(losses)
        final_loss = losses[-1][1] if losses else float("inf")
        _log(
            f"  slope={slope:.6f}  final_loss={final_loss:.4f}  "
            f"points={len(losses)}  {elapsed:.1f}s"
        )
        results[rank] = {
            "slope": slope, "final_loss": final_loss,
            "n_points": len(losses), "time_s": elapsed, "ok": True,
        }
        yaml_path.unlink(missing_ok=True)

    # ── Select best rank ─────────────────────────────────────────────────
    ok_results = {r: d for r, d in results.items() if d.get("ok")}
    if not ok_results:
        _log("WARNING: all probes failed — falling back to rank 32")
        best = 32
    else:
        slopes_abs = {r: abs(d["slope"]) for r, d in ok_results.items()}
        steepest = max(slopes_abs.values())
        threshold = (1.0 - tolerance) * steepest

        _log(f"steepest |slope| = {steepest:.6f},  threshold ({tolerance*100:.0f}%) = {threshold:.6f}")

        best = max(ranks)  # fallback: largest rank
        for rank in sorted(ok_results.keys()):
            if slopes_abs[rank] >= threshold:
                best = rank
                break

    _log(f">>> selected rank = {best}")

    # ── Write report ─────────────────────────────────────────────────────
    report_path = ROOT / "saves" / "_probes" / f"{tag}_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_report(report_path, tag, trunk_path, dataset_name, steps, tolerance, best, results)
    _log(f"report: {report_path}")

    # ── Cleanup ──────────────────────────────────────────────────────────
    if cleanup:
        for d in probe_dirs:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
        _log("cleaned up probe output dirs")

    return best, results


def _write_report(
    path: Path,
    tag: str,
    trunk_path: str,
    dataset_name: str,
    steps: int,
    tolerance: float,
    best: int,
    results: dict[int, dict],
) -> None:
    """Write a Markdown probe report."""
    lines = [
        f"# Rank Probe Report — `{tag}`",
        "",
        f"| Key | Value |",
        f"|-----|-------|",
        f"| Trunk | `{trunk_path}` |",
        f"| Dataset | `{dataset_name}` |",
        f"| Steps/probe | {steps} |",
        f"| Tolerance | {tolerance * 100:.0f}% |",
        f"| **Selected rank** | **{best}** |",
        "",
        "## Per-rank results",
        "",
        "| Rank | Slope | |Slope| | Final Loss | Points | Time (s) | Status |",
        "|-----:|------:|-------:|-----------:|-------:|---------:|--------|",
    ]
    for rank in sorted(results.keys()):
        d = results[rank]
        if rank == best:
            status = "**SELECTED**"
        elif d.get("ok"):
            status = "ok"
        else:
            status = "FAILED"
        lines.append(
            f"| {rank} | {d['slope']:.6f} | {abs(d['slope']):.6f} "
            f"| {d['final_loss']:.4f} | {d['n_points']} | {d['time_s']:.1f} | {status} |"
        )
    lines += [
        "",
        "## Selection rule",
        "",
        "Pick the **lowest** rank whose |slope| is within the tolerance "
        "threshold of the steepest |slope| observed.  This finds the smallest",
        "adapter that still learns effectively, avoiding both under-capacity",
        "(too small → slow convergence) and over-capacity (too large → wasted",
        "parameters, possible overfitting).",
        "",
        "*Generated by `scripts/rank_probe.py`*",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rank Probe — empirical LoRA rank selection via short training sweeps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Probe for a skill dataset already registered in data/dataset_info.json
  %(prog)s --trunk saves/trunk_v1/merged \\
           --dataset trunk_v1_translate_branch \\
           --tag probe_translate

  # Custom ranks and more steps
  %(prog)s --trunk saves/trunk_v1/merged \\
           --dataset my_ds \\
           --tag probe_custom \\
           --ranks 8,16,32,64,128,256 \\
           --steps 100
""",
    )
    parser.add_argument("--trunk", required=True,
                        help="Path to merged trunk model (HF dir)")
    parser.add_argument("--dataset", required=True,
                        help="Dataset name registered in data/dataset_info.json")
    parser.add_argument("--tag", required=True,
                        help="Unique tag for this probe run")
    parser.add_argument("--ranks", default="16,32,64,128",
                        help="Comma-separated candidate ranks (default: 16,32,64,128)")
    parser.add_argument("--steps", type=int, default=PROBE_STEPS,
                        help=f"Training steps per probe (default {PROBE_STEPS})")
    parser.add_argument("--tolerance", type=float, default=SLOPE_TOLERANCE,
                        help=f"Slope tolerance 0-1 (default {SLOPE_TOLERANCE})")
    parser.add_argument("--cpu-safe", action="store_true",
                        help="CPU-friendly config (no bf16, adamw_torch)")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Keep probe output dirs for inspection")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate probe configs but skip training")

    args = parser.parse_args()

    trunk = Path(args.trunk).resolve()
    if not (trunk / "config.json").exists():
        _log(f"ERROR: trunk {trunk} has no config.json — is it a merged HF model?")
        return 1

    ranks = [int(r.strip()) for r in args.ranks.split(",") if r.strip()]
    if not ranks:
        _log("ERROR: no ranks specified")
        return 1

    trunk_str = str(trunk).replace("\\", "/")

    # ── Dry run: just write configs ──────────────────────────────────────
    if args.dry_run:
        for rank in ranks:
            cfg = _probe_yaml(trunk_str, args.dataset, rank, args.tag, args.cpu_safe, args.steps)
            yaml_path = (
                ROOT / "examples" / "distillation" / "auto" / f"_probe_{args.tag}_r{rank}.yaml"
            )
            yaml_path.parent.mkdir(parents=True, exist_ok=True)
            with yaml_path.open("w", encoding="utf-8") as f:
                f.write(f"### Probe config: rank={rank}\n\n")
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            _log(f"wrote {yaml_path}")
        _log("--dry-run: configs written, stopping before training")
        return 0

    # ── Full probe run ───────────────────────────────────────────────────
    best, results = probe_rank(
        trunk_path=trunk_str,
        dataset_name=args.dataset,
        tag=args.tag,
        cpu_safe=args.cpu_safe,
        ranks=ranks,
        steps=args.steps,
        tolerance=args.tolerance,
        cleanup=not args.no_cleanup,
    )

    _log(f"RESULT: optimal rank = {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
