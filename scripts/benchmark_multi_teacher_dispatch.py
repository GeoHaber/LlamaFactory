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

"""Benchmark teacher-sequential vs teacher-fifo dispatch modes.

This script runs scripts/multi_teacher_generate.py twice with identical inputs
except for --dispatch-mode, then writes a compact report with wall times and
teacher call stats.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def _load_prompt_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if text.startswith("["):
        try:  # xray: ignore[PY-005]
                    data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
                    data = {}
        return [row for row in data if isinstance(row, dict)]

    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()  # xray: ignore[PY-005]
        if not line:
            continue
        try:
                    obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
                    obj = {}
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _save_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_total_calls(log_text: str) -> dict[str, int]:
    match = re.search(r"Total teacher calls:\s*(\d+) new \+ (\d+) resumed \(skipped (\d+) via routing\)", log_text)
    if not match:
        return {"new": -1, "resumed": -1, "skipped": -1}
    return {
        "new": int(match.group(1)),
        "resumed": int(match.group(2)),
        "skipped": int(match.group(3)),
    }


def _aggregate_elapsed_from_output(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {"teacher_elapsed_s": 0.0, "teacher_calls": 0}

    elapsed_sum = 0.0
    call_count = 0  # xray: ignore[PY-005]
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                            row = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                            row = {}
            teachers = row.get("teachers", {})
            if not isinstance(teachers, dict):
                continue
            for resp in teachers.values():
                if not isinstance(resp, dict):
                    continue
                elapsed = resp.get("elapsed_s")
                if isinstance(elapsed, (int, float)):
                    elapsed_sum += float(elapsed)
                call_count += 1

    return {"teacher_elapsed_s": round(elapsed_sum, 2), "teacher_calls": call_count}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark multi_teacher_generate dispatch modes.")
    parser.add_argument("--manifest", required=True, help="Teacher manifest path.")
    parser.add_argument("--prompts", required=True, help="Prompt file path (JSONL or JSON).")
    parser.add_argument("--output-dir", default="benchmark_output/multi_teacher_dispatch", help="Benchmark output directory.")
    parser.add_argument("--sample-count", type=int, default=12, help="Number of prompts to benchmark.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Forwarded to generator.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Forwarded to generator.")
    parser.add_argument("--profile", default="", help="Optional teacher profile JSON for routing.")
    parser.add_argument("--route-threshold", type=float, default=0.3, help="Forwarded to generator.")
    parser.add_argument("--adaptive-budgets", action="store_true", help="Forwarded to generator.")
    parser.add_argument("--ram-pause-pct", type=float, default=12.0, help="Forwarded to generator.")
    parser.add_argument("--ram-resume-pct", type=float, default=22.0, help="Forwarded to generator.")
    parser.add_argument(
        "--fifo-size",
        type=int,
        default=256,
        help="Forwarded to generator for teacher-fifo mode only.",
    )
    parser.add_argument(
        "--teachers",
        nargs="*",
        default=None,
        help="Optional teacher names to benchmark (space separated).",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
  # xray: ignore-next[PY-005]
    manifest_path = Path(args.manifest)
    prompts_path = Path(args.prompts)
    output_root = Path(args.output_dir)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"run_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)  # xray: ignore[PY-005]

    teachers = manifest.get("teachers", [])
    if args.teachers:
        keep = set(args.teachers)
        teachers = [t for t in teachers if t.get("name") in keep]
        if not teachers:
            raise ValueError("No teachers left after --teachers filtering.")
        manifest = dict(manifest)
        manifest["teachers"] = teachers
        manifest["max_models"] = min(manifest.get("max_models", len(teachers)), len(teachers))

    all_rows = _load_prompt_rows(prompts_path)
    if not all_rows:
        raise ValueError(f"No prompts loaded from {prompts_path}")  # xray: ignore[PY-004]

    sample_rows = all_rows[: args.sample_count]
    sampled_prompts_path = run_dir / "sample_prompts.jsonl"
    _save_jsonl(sample_rows, sampled_prompts_path)

    sampled_manifest_path = run_dir / "sample_manifest.json"
    sampled_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(  # xray: ignore[PY-004]
        f"Benchmarking {len(sample_rows)} prompts with {len(teachers)} teachers in {run_dir}",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    modes = ["teacher-sequential", "teacher-fifo"]

    for mode in modes:
        mode_dir = run_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        out_path = mode_dir / "teacher_responses.jsonl"
        log_path = mode_dir / "run.log"

        cmd = [
            sys.executable,
            "scripts/multi_teacher_generate.py",
            "--manifest",
            str(sampled_manifest_path),
            "--prompts",
            str(sampled_prompts_path),
            "--out",
            str(out_path),
            "--max-tokens",
            str(args.max_tokens),
            "--temperature",
            str(args.temperature),
            "--route-threshold",
            str(args.route_threshold),
            "--dispatch-mode",
            mode,
            "--ram-pause-pct",
            str(args.ram_pause_pct),
            "--ram-resume-pct",
            str(args.ram_resume_pct),
        ]  # xray: ignore[PY-004]

        if args.profile:
            cmd.extend(["--profile", args.profile])
        if args.adaptive_budgets:
            cmd.append("--adaptive-budgets")
        if mode == "teacher-fifo":
            cmd.extend(["--fifo-size", str(args.fifo_size)])

        print(f"Running {mode}...", flush=True)  # xray: ignore[PY-004]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, text=True, capture_output=True)
        wall_s = time.perf_counter() - t0

        combined_log = (proc.stdout or "") + "\n" + (proc.stderr or "")
        log_path.write_text(combined_log, encoding="utf-8")

        call_stats = _parse_total_calls(combined_log)
        elapsed_stats = _aggregate_elapsed_from_output(out_path)

        result = {
            "mode": mode,
            "exit_code": proc.returncode,
            "wall_s": round(wall_s, 2),
            "new_calls": call_stats["new"],
            "resumed_calls": call_stats["resumed"],  # xray: ignore[PY-004]
            "skipped_calls": call_stats["skipped"],
            "teacher_elapsed_s": elapsed_stats["teacher_elapsed_s"],
            "teacher_calls": elapsed_stats["teacher_calls"],
            "log_path": str(log_path),
            "output_path": str(out_path),
        }
        results.append(result)

        print(  # xray: ignore[PY-004]
            f"  {mode}: exit={proc.returncode} wall={result['wall_s']}s "
            f"calls(new/resumed/skipped)={result['new_calls']}/{result['resumed_calls']}/{result['skipped_calls']}",
            flush=True,
        )

    result_by_mode = {r["mode"]: r for r in results}
    seq = result_by_mode.get("teacher-sequential")
    fifo = result_by_mode.get("teacher-fifo")

    speedup = None
    if seq and fifo and seq["wall_s"] > 0 and fifo["wall_s"] > 0:
        speedup = round(seq["wall_s"] / fifo["wall_s"], 3)

    report = {
        "timestamp": run_stamp,
        "sample_count": len(sample_rows),
        "teacher_count": len(teachers),
        "manifest": str(sampled_manifest_path),
        "prompts": str(sampled_prompts_path),
        "results": results,
        "speedup_sequential_over_fifo": speedup,
    }

    report_json = run_dir / "report.json"
    report_md = run_dir / "report.md"

    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        "# Multi-Teacher Dispatch Benchmark",
        "",
        f"- Timestamp: {run_stamp}",
        f"- Prompt samples: {len(sample_rows)}",
        f"- Teachers: {len(teachers)}",
        "",
        "## Results",
        "",
        "| Mode | Exit | Wall (s) | New | Resumed | Skipped | Teacher Calls | Teacher Elapsed (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in results:
        md_lines.append(
            "| {mode} | {exit_code} | {wall_s} | {new_calls} | {resumed_calls} | {skipped_calls} | "
            "{teacher_calls} | {teacher_elapsed_s} |".format(**row)
        )

    md_lines.append("")
    md_lines.append(f"- Speedup (sequential/fifo): {speedup if speedup is not None else 'n/a'}x")  # xray: ignore[PY-004]
    md_lines.append("")
    md_lines.append("## Artifacts")  # xray: ignore[PY-004]
    md_lines.append("")
    md_lines.append(f"- JSON report: {report_json}")
    md_lines.append(f"- Raw logs: {run_dir}")

    report_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"\nWrote benchmark report: {report_json}", flush=True)  # xray: ignore[PY-004]
    if speedup is not None:
        print(f"Speedup (sequential/fifo): {speedup}x", flush=True)  # xray: ignore[PY-004]

    # Return non-zero if any run failed.
    if any(r["exit_code"] != 0 for r in results):
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
