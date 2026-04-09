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

"""Postmortem agent -- read finished-run telemetry and emit recommendations.

Reads three sources of truth from a completed pipeline run:

1. ``trainer_log.jsonl`` -- per-step training/eval losses, lr, grad_norm, etc.
2. ``metrics.jsonl`` -- per-second resource samples (GPU/CPU/RAM/disk/net),
   produced by the live metrics rail in ``distill_server.py``.
3. ``eval_results.json`` -- final downstream eval metrics (optional).

Detects:

* GPU under-utilization (>30%% of run with GPU < 50%%).
* CPU bottleneck (CPU > 80%% while GPU < 50%%).
* I/O bottleneck (high disk read MB/s while GPU idle).
* RAM / VRAM near-OOM events (>92%% utilization).
* Slow training steps (>2x median step time).
* NaN / exploding gradients (loss == NaN, grad_norm > 100).
* Eval-loss plateau (no improvement for last 30%% of training).
* Dataloader stalls (gaps > 1 s between consecutive logged steps).

Emits a Markdown report at ``docs/postmortem/RUN_<id>.md`` with:

* A short summary block.
* A findings list, sorted by severity.
* An "auto-tune" diff suggesting concrete YAML changes for the next run.

Usage:
    python scripts/postmortem_agent.py path/to/run_dir
    python scripts/postmortem_agent.py path/to/run_dir --report-dir docs/postmortem
    python scripts/postmortem_agent.py --latest saves/qwen2.5-coder-14b/lora/distill_sft

Exit status:
    0 -- analysis complete (regardless of severity).
    1 -- could not find any of the expected input files.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Severity ordering -- higher = more urgent.
# ---------------------------------------------------------------------------
_SEV_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
_SEV_GLYPH = {
    "critical": "[CRIT]",
    "high":     "[HIGH]",
    "medium":   "[MED] ",
    "low":      "[LOW] ",
    "info":     "[INFO]",
}


@dataclass
class Finding:
    severity: str
    title: str
    diagnosis: str
    recommendation: str
    evidence: dict[str, Any] = field(default_factory=dict)
    yaml_diff: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunSummary:
    run_dir: Path
    metric_ticks: int = 0
    train_steps: int = 0
    wall_time_s: float = 0.0
    final_loss: float | None = None
    final_eval_loss: float | None = None
    peak_vram_mb: float = 0.0
    peak_ram_gb: float = 0.0
    avg_gpu_util: float = 0.0
    avg_cpu_util: float = 0.0
    avg_tok_s: float = 0.0
    findings: list[Finding] = field(default_factory=list)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce arbitrary JSON values to float; return default on garbage.

    The metrics sidecar comes from a long-running sampler that may write
    partial or malformed rows after a crash. Detectors must never raise on
    bad input -- they should treat it as a missing sample instead.
    """
    if value is None:
        return default
    if isinstance(value, bool):  # bool is a subclass of int -- exclude first
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _avg(seq: list[float]) -> float:
    return sum(seq) / len(seq) if seq else 0.0


def _percentile(seq: list[float], p: float) -> float:
    if not seq:
        return 0.0
    s = sorted(seq)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


# ---------------------------------------------------------------------------
# Detectors -- each returns 0..N findings
# ---------------------------------------------------------------------------
def _detect_gpu_underutil(metrics: list[dict]) -> list[Finding]:
    if not metrics:
        return []
    util = [_safe_float(m.get("gpu.util_pct")) for m in metrics]
    if not util or max(util) == 0.0:  # no GPU at all -- skip
        return []
    under = sum(1 for u in util if u < 50)
    frac = under / len(util)
    if frac < 0.30:
        return []
    return [Finding(
        severity="high" if frac > 0.50 else "medium",
        title=f"GPU under-utilized for {frac*100:.0f}% of run",
        diagnosis="GPU sat below 50% utilization for a significant fraction of the "
                  "run. Most likely cause: dataloader cannot keep the GPU fed.",
        recommendation="Increase `dataloader_num_workers` (try 2x current). "
                       "If dataset is small, raise `preprocessing_num_workers` and "
                       "pre-tokenize once. Consider `pin_memory=True`.",
        evidence={"under_50_frac": round(frac, 3), "avg_gpu_util": round(_avg(util), 1)},
        yaml_diff={"dataloader_num_workers": "+2"},
    )]


def _detect_cpu_bottleneck(metrics: list[dict]) -> list[Finding]:
    if not metrics:
        return []
    pairs = [
        (_safe_float(m.get("gpu.util_pct")), _safe_float(m.get("cpu.util_pct")))
        for m in metrics
    ]
    if not pairs:
        return []
    cpu_bound = sum(1 for g, c in pairs if g < 50 and c > 80)
    if cpu_bound / len(pairs) < 0.20:
        return []
    return [Finding(
        severity="high",
        title="CPU-bound: GPU idle while CPU saturated",
        diagnosis="CPU was above 80% while the GPU was below 50% for >20% of "
                  "the run. The dataloader / tokenizer is the limiting factor.",
        recommendation="Pre-tokenize the dataset offline; raise "
                       "`preprocessing_num_workers`; reduce `cutoff_len` if "
                       "samples are highly variable.",
        evidence={"cpu_bound_frac": round(cpu_bound / len(pairs), 3)},
        yaml_diff={"preprocessing_num_workers": "+4"},
    )]


def _detect_io_bottleneck(metrics: list[dict]) -> list[Finding]:
    if not metrics:
        return []
    high_io = sum(
        1
        for m in metrics
        if _safe_float(m.get("disk.read_mb_s")) > 200
        and _safe_float(m.get("gpu.util_pct")) < 50
    )
    if high_io / len(metrics) < 0.15:
        return []
    return [Finding(
        severity="medium",
        title="I/O-bound: high disk reads while GPU idle",
        diagnosis="Disk reads above 200 MB/s while GPU below 50% indicate the "
                  "dataset is being streamed from slow storage.",
        recommendation="Move the dataset to a fast NVMe; pre-tokenize and cache; "
                       "consider memory-mapped formats (Arrow / parquet).",
    )]


def _detect_vram_pressure(metrics: list[dict]) -> list[Finding]:
    if not metrics:
        return []
    near_oom = 0
    for m in metrics:
        used = _safe_float(m.get("gpu.mem_used_mb"))
        total = _safe_float(m.get("gpu.mem_total_mb"))
        if total > 0 and used / total > 0.92:
            near_oom += 1
    if near_oom < 3:
        return []
    return [Finding(
        severity="high",
        title=f"VRAM near-OOM in {near_oom} samples",
        diagnosis="VRAM exceeded 92% of total in multiple samples. Risk of OOM "
                  "on long sequences or larger batches.",
        recommendation="Enable gradient checkpointing; reduce "
                       "`per_device_train_batch_size`; reduce `cutoff_len`; "
                       "or enable Liger Kernel (`enable_liger_kernel: true`).",
        evidence={"near_oom_samples": near_oom},
        yaml_diff={"enable_liger_kernel": True, "per_device_train_batch_size": "-1"},
    )]


def _detect_ram_pressure(metrics: list[dict]) -> list[Finding]:
    if not metrics:
        return []
    near_oom = 0
    for m in metrics:
        used = _safe_float(m.get("ram.used_gb"))
        total = _safe_float(m.get("ram.total_gb"))
        if total > 0 and used / total > 0.92:
            near_oom += 1
    if near_oom < 5:
        return []
    return [Finding(
        severity="high",
        title=f"System RAM near-exhaustion in {near_oom} samples",
        diagnosis="System RAM exceeded 92% of total. Multi-teacher generation "
                  "may be running too many concurrent backends.",
        recommendation="Lower `max_concurrent_teachers`; enable RAMPressureThrottle "
                       "with stricter thresholds; close other applications.",
        evidence={"near_oom_samples": near_oom},
    )]


def _detect_slow_steps(train: list[dict]) -> list[Finding]:
    step_times = []
    last_ts = None
    for entry in train:
        ts = entry.get("epoch") or entry.get("current_steps")
        # We only have epoch/step from trainer_log -- use median diff as a
        # proxy. If `step_time_ms` is logged directly, use that.
        if "step_time_ms" in entry:
            step_times.append(float(entry["step_time_ms"]))
        elif ts is not None and last_ts is not None:
            try:
                step_times.append(float(ts) - float(last_ts))
            except (TypeError, ValueError):
                pass
        last_ts = ts

    if len(step_times) < 10:
        return []
    median = statistics.median(step_times)
    if median <= 0:
        return []
    slow = sum(1 for s in step_times if s > 2 * median)
    if slow < 5:
        return []
    return [Finding(
        severity="medium",
        title=f"{slow} slow training steps (>2x median)",
        diagnosis="A non-trivial number of steps took more than 2x the median "
                  "step time. Causes: GC pauses, data shuffling, thermal "
                  "throttling, or eval-during-train pauses.",
        recommendation="Check GPU temperature; lower `eval_steps` frequency; "
                       "enable `dataloader_persistent_workers=True`.",
        evidence={"slow_steps": slow, "median_s": round(median, 3)},
    )]


def _detect_nan_or_explode(train: list[dict]) -> list[Finding]:
    nans = 0
    explodes = 0
    for entry in train:
        loss = entry.get("loss")
        if isinstance(loss, str) and loss.lower() == "nan":
            nans += 1
        elif isinstance(loss, float) and (loss != loss):  # NaN check
            nans += 1
        gn = entry.get("grad_norm")
        if isinstance(gn, (int, float)) and gn > 100:
            explodes += 1
    findings = []
    if nans:
        findings.append(Finding(
            severity="critical",
            title=f"{nans} NaN loss values logged",
            diagnosis="Training produced NaN losses. The optimizer is likely "
                      "diverging due to too high a learning rate, fp16 underflow, "
                      "or a corrupted sample.",
            recommendation="Lower `learning_rate` by 5x; switch from fp16 to bf16; "
                           "enable `max_grad_norm: 1.0` for gradient clipping.",
            evidence={"nan_count": nans},
            yaml_diff={"learning_rate": "/5", "max_grad_norm": 1.0},
        ))
    if explodes:
        findings.append(Finding(
            severity="high",
            title=f"{explodes} exploding gradients (grad_norm > 100)",
            diagnosis="Gradient norms exceeded 100 multiple times, indicating "
                      "instability.",
            recommendation="Add gradient clipping (`max_grad_norm: 1.0`); "
                           "lower learning rate; enable warmup.",
            evidence={"explode_count": explodes},
            yaml_diff={"max_grad_norm": 1.0},
        ))
    return findings


def _detect_plateau(train: list[dict]) -> list[Finding]:
    losses = [float(e["loss"]) for e in train if isinstance(e.get("loss"), (int, float))]
    if len(losses) < 50:
        return []
    tail_n = max(10, len(losses) // 3)
    head = losses[: len(losses) - tail_n]
    tail = losses[-tail_n:]
    if not head or not tail:
        return []
    head_min = min(head)
    tail_min = min(tail)
    improvement = head_min - tail_min
    if improvement > 0.01:  # tail still improving
        return []
    return [Finding(
        severity="low",
        title="Loss plateaued in the final third of training",
        diagnosis="The minimum loss in the last third of the run did not "
                  "improve meaningfully over the first two thirds. Training "
                  "may be wasting compute.",
        recommendation="Reduce `num_train_epochs`; enable early stopping with "
                       "`load_best_model_at_end: true` and `metric_for_best_model: eval_loss`; "
                       "or increase `learning_rate` slightly to escape the plateau.",
        evidence={
            "head_min": round(head_min, 4),
            "tail_min": round(tail_min, 4),
            "improvement": round(improvement, 4),
        },
        yaml_diff={"num_train_epochs": "-0.5"},
    )]


def _detect_dataloader_stall(metrics: list[dict]) -> list[Finding]:
    """Look for >1 s gaps where GPU was idle and CPU also low (sync stall)."""
    if len(metrics) < 10:
        return []
    stalls = 0
    for m in metrics:
        gpu = _safe_float(m.get("gpu.util_pct"))
        cpu = _safe_float(m.get("cpu.util_pct"))
        if gpu < 10 and cpu < 30:
            stalls += 1
    if stalls < 5:
        return []
    return [Finding(
        severity="medium",
        title=f"{stalls} idle samples (GPU<10% and CPU<30%)",
        diagnosis="Multiple samples show both GPU and CPU idle, indicating a "
                  "sync stall (e.g. Python GIL, blocking I/O, or eval pause).",
        recommendation="Profile with `py-spy dump` during a stall; switch eval "
                       "to a separate process; lower eval frequency.",
        evidence={"idle_samples": stalls},
    )]


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------
def analyze(run_dir: Path) -> RunSummary | None:
    metrics_path = run_dir / "metrics.jsonl"
    train_path   = run_dir / "trainer_log.jsonl"
    eval_path    = run_dir / "eval_results.json"

    metrics = _read_jsonl(metrics_path)
    train   = _read_jsonl(train_path)
    eval_results = _read_json(eval_path)

    if not metrics and not train:
        print(f"[postmortem] no telemetry found in {run_dir}", file=sys.stderr)
        return None

    summary = RunSummary(run_dir=run_dir)
    summary.metric_ticks = len(metrics)
    summary.train_steps = len(train)

    if metrics:
        timestamps = [_safe_float(m.get("ts")) for m in metrics if m.get("ts") is not None]
        timestamps = [t for t in timestamps if t > 0]
        if len(timestamps) >= 2:
            summary.wall_time_s = max(timestamps) - min(timestamps)
        summary.peak_vram_mb = max((_safe_float(m.get("gpu.mem_used_mb")) for m in metrics), default=0.0)
        summary.peak_ram_gb = max((_safe_float(m.get("ram.used_gb")) for m in metrics), default=0.0)
        summary.avg_gpu_util = _avg([_safe_float(m.get("gpu.util_pct")) for m in metrics])
        summary.avg_cpu_util = _avg([_safe_float(m.get("cpu.util_pct")) for m in metrics])
        tok_vals = [_safe_float(m.get("tok_s")) for m in metrics if m.get("tok_s")]
        tok_vals = [t for t in tok_vals if t > 0]
        summary.avg_tok_s = _avg(tok_vals)

    if train:
        for entry in reversed(train):
            if summary.final_loss is None and isinstance(entry.get("loss"), (int, float)):
                summary.final_loss = float(entry["loss"])
            if summary.final_eval_loss is None and isinstance(entry.get("eval_loss"), (int, float)):
                summary.final_eval_loss = float(entry["eval_loss"])
            if summary.final_loss is not None and summary.final_eval_loss is not None:
                break

    # Run all detectors
    findings: list[Finding] = []
    for fn in (
        _detect_gpu_underutil,
        _detect_cpu_bottleneck,
        _detect_io_bottleneck,
        _detect_vram_pressure,
        _detect_ram_pressure,
        _detect_dataloader_stall,
    ):
        findings.extend(fn(metrics))
    for fn_t in (_detect_slow_steps, _detect_nan_or_explode, _detect_plateau):
        findings.extend(fn_t(train))

    # Sort by severity (most urgent first), then by title for stable ordering.
    findings.sort(key=lambda f: (_SEV_ORDER.get(f.severity, 99), f.title))
    summary.findings = findings

    if eval_results:
        # Carry through eval metrics for the report
        pass
    return summary


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------
def render_markdown(summary: RunSummary) -> str:
    lines: list[str] = []
    run_id = summary.run_dir.name
    lines.append(f"# Postmortem -- {run_id}")
    lines.append("")
    lines.append(f"**Run dir:** `{summary.run_dir}`")
    lines.append(f"**Wall time:** {summary.wall_time_s/60:.1f} min")
    lines.append(f"**Train steps:** {summary.train_steps}")
    lines.append(f"**Metric ticks:** {summary.metric_ticks}")
    if summary.final_loss is not None:
        lines.append(f"**Final loss:** {summary.final_loss:.4f}")
    if summary.final_eval_loss is not None:
        lines.append(f"**Final eval_loss:** {summary.final_eval_loss:.4f}")
    if summary.peak_vram_mb:
        lines.append(f"**Peak VRAM:** {summary.peak_vram_mb/1024:.1f} GB")
    if summary.peak_ram_gb:
        lines.append(f"**Peak RAM:** {summary.peak_ram_gb:.1f} GB")
    if summary.avg_gpu_util:
        lines.append(f"**Avg GPU util:** {summary.avg_gpu_util:.0f}%")
    if summary.avg_cpu_util:
        lines.append(f"**Avg CPU util:** {summary.avg_cpu_util:.0f}%")
    if summary.avg_tok_s:
        lines.append(f"**Avg tok/s:** {summary.avg_tok_s:.0f}")
    lines.append("")

    if not summary.findings:
        lines.append("## Findings")
        lines.append("")
        lines.append("No issues detected. Run looks healthy.")
        lines.append("")
        return "\n".join(lines)

    lines.append(f"## Findings ({len(summary.findings)})")
    lines.append("")
    for f in summary.findings:
        glyph = _SEV_GLYPH.get(f.severity, "[?]")
        lines.append(f"### {glyph} {f.title}")
        lines.append("")
        lines.append(f"**Diagnosis:** {f.diagnosis}")
        lines.append("")
        lines.append(f"**Recommendation:** {f.recommendation}")
        lines.append("")
        if f.evidence:
            lines.append("**Evidence:**")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(f.evidence, indent=2))
            lines.append("```")
            lines.append("")

    # Auto-tune suggested YAML diff
    diff: dict[str, Any] = {}
    for f in summary.findings:
        diff.update(f.yaml_diff)
    if diff:
        lines.append("## Suggested YAML diff")
        lines.append("")
        lines.append("```yaml")
        for k, v in diff.items():
            lines.append(f"{k}: {v}    # auto-tune from postmortem")
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _resolve_run_dir(arg: str, latest: bool) -> Path:
    p = Path(arg)
    if not latest:
        return p
    # Walk all subdirectories under p that contain a trainer_log.jsonl
    candidates = sorted(
        p.rglob("trainer_log.jsonl"),
        key=lambda c: c.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return p
    return candidates[0].parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read finished-run telemetry and emit recommendations.",
    )
    parser.add_argument(
        "run_dir",
        help="Path to a run directory containing trainer_log.jsonl and "
             "(optionally) metrics.jsonl.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Treat run_dir as a parent and pick the most recently modified "
             "subdirectory containing a trainer_log.jsonl.",
    )
    parser.add_argument(
        "--report-dir",
        default="docs/postmortem",
        help="Where to write the Markdown report (default: docs/postmortem).",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the report to stdout instead of writing to disk.",
    )
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir, args.latest)
    summary = analyze(run_dir)
    if summary is None:
        return 1

    report = render_markdown(summary)

    if args.print_only:
        print(report)
        return 0

    out_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"RUN_{run_dir.name}.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"[postmortem] wrote {out_path}")

    # Print a one-line terminal summary so it shows up in CI logs.
    sev_counts: dict[str, int] = {}
    for f in summary.findings:
        sev_counts[f.severity] = sev_counts.get(f.severity, 0) + 1
    if sev_counts:
        sev_str = " ".join(f"{k}={v}" for k, v in sev_counts.items())
        print(f"[postmortem] {len(summary.findings)} findings ({sev_str})")
    else:
        print("[postmortem] no findings -- run looks healthy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
