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

"""Student Registry — longitudinal CRUD tracker for trained student models.

Stores per-student metadata, eval history, lineage, and gap analysis.
Registry lives at saves/student_registry.json (gitignored).
Use --export-registry to dump a human-readable Markdown report.

Usage:
    # Register a new student (auto-fills from forge_results + eval_scorecards)
    python scripts/student_registry.py register \\
        --saves-tag zena007 --variant-id B

    # List all students
    python scripts/student_registry.py list

    # Update eval history with new scores
    python scripts/student_registry.py update-eval \\
        --model-id zena007/B --eval-file saves/zena007/eval_scorecards.jsonl

    # Gap analysis for a student
    python scripts/student_registry.py gap-analysis --model-id zena007/B

    # Compare two students
    python scripts/student_registry.py compare \\
        --model-ids zena007/B zena007/C

    # Export Markdown report
    python scripts/student_registry.py export --output student_report.md
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REGISTRY_PATH = Path("saves/student_registry.json")


# ── Registry I/O ─────────────────────────────────────────────────────────

def _load_registry() -> dict:
    """Load or initialize the registry."""
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))  # xray: ignore[PY-005]
    return {"version": 1, "students": {}}


def _save_registry(reg: dict) -> None:
    """Write registry atomically."""
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = REGISTRY_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(REGISTRY_PATH)


# ── Commands ─────────────────────────────────────────────────────────────

def cmd_register(args: argparse.Namespace) -> int:
    """Register a student from forge results."""
    tag = args.saves_tag
    vid = args.variant_id
    model_id = f"{tag}/{vid}"

    forge_path = Path(f"saves/{tag}/forge_results.jsonl")
    if not forge_path.exists():
        print(f"ERROR: Forge results not found: {forge_path}")  # xray: ignore[PY-004]
        return 1

    # Find variant in forge results
    variant_data = None
    for line in forge_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:  # xray: ignore[PY-005]
                    r = json.loads(line)
        except (json.JSONDecodeError, ValueError):
                    r = {}
        if r.get("variant_id") == vid:
            variant_data = r
            break  # xray: ignore[PY-004]

    if variant_data is None:
        print(f"ERROR: Variant {vid} not found in {forge_path}")  # xray: ignore[PY-004]
        return 1

    # Load eval scorecard if available
    eval_entry = None
    scorecard_path = Path(f"saves/{tag}/eval_scorecards.jsonl")
    if scorecard_path.exists():
        for line in scorecard_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            sc = json.loads(line)  # xray: ignore[PY-005]
            if sc.get("variant_id") == vid:
                eval_entry = sc
                break

    reg = _load_registry()
    now = datetime.now(timezone.utc).isoformat()

    entry = {
        "model_id": model_id,
        "status": "trained",
        "created": now,
        "updated": now,
        "lineage": {
            "base_model": variant_data.get("model", ""),
            "sft_adapter": variant_data.get("sft_adapter_path", ""),
            "dpo_adapter": variant_data.get("dpo_adapter_path", ""),
            "data_tag": tag,
        },
        "training_config": {
            "lora_rank": variant_data.get("lora_rank"),
            "learning_rate": variant_data.get("learning_rate"),
            "dpo_enabled": variant_data.get("dpo_enabled", False),
        },
        "loss_history": {
            "sft_final_loss": variant_data.get("sft_final_loss"),
            "dpo_final_loss": variant_data.get("dpo_final_loss"),
        },
        "eval_history": [],
        "gguf_variants": [],
    }

    if eval_entry:
        entry["eval_history"].append({
            "timestamp": now,
            "avg_loss": eval_entry.get("avg_loss"),
            "avg_ppl": eval_entry.get("avg_ppl"),
            "score": eval_entry.get("score"),
            "category_scores": eval_entry.get("category_scores", {}),
        })
        if eval_entry.get("score") is not None:
            entry["status"] = "evaluated"
  # xray: ignore-next[PY-004]
    reg["students"][model_id] = entry
    _save_registry(reg)
    print(f"Registered: {model_id} (status={entry['status']})")  # xray: ignore[PY-004]
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List all registered students."""
    reg = _load_registry()
    students = reg.get("students", {})  # xray: ignore[PY-004]

    if not students:
        print("Registry is empty.")  # xray: ignore[PY-004]
        return 0  # xray: ignore[PY-004]

    print(f"{'Model ID':<25} {'Status':<12} {'SFT Loss':>9} {'Score':>7} {'Created'}")  # xray: ignore[PY-004]
    print("-" * 80)  # xray: ignore[PY-004]
    for mid, s in sorted(students.items()):
        sft = s.get("loss_history", {}).get("sft_final_loss")
        sft_str = f"{sft:.4f}" if sft is not None else "n/a"
        score = None
        if s.get("eval_history"):
            score = s["eval_history"][-1].get("score")  # xray: ignore[PY-004]
        score_str = f"{score:.4f}" if score is not None else "n/a"
        created = s.get("created", "")[:10]
        print(f"{mid:<25} {s.get('status', '?'):<12} {sft_str:>9} {score_str:>7} {created}")  # xray: ignore[PY-004]

    return 0


def cmd_update_eval(args: argparse.Namespace) -> int:
    """Append a new eval snapshot to a student's history."""
    model_id = args.model_id
    eval_file = Path(args.eval_file)  # xray: ignore[PY-004]

    if not eval_file.exists():
        print(f"ERROR: Eval file not found: {eval_file}")  # xray: ignore[PY-004]
        return 1
  # xray: ignore-next[PY-004]
    reg = _load_registry()
    if model_id not in reg["students"]:
        print(f"ERROR: Student {model_id} not in registry.")  # xray: ignore[PY-004]
        return 1

    # Find entry for this variant
    vid = model_id.split("/")[-1]
    eval_entry = None
    for line in eval_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sc = json.loads(line)  # xray: ignore[PY-005]
        if sc.get("variant_id") == vid:
            eval_entry = sc
            break  # xray: ignore[PY-004]

    if eval_entry is None:
        print(f"ERROR: No scorecard for variant {vid} in {eval_file}")  # xray: ignore[PY-004]
        return 1

    now = datetime.now(timezone.utc).isoformat()
    reg["students"][model_id]["eval_history"].append({
        "timestamp": now,
        "avg_loss": eval_entry.get("avg_loss"),
        "avg_ppl": eval_entry.get("avg_ppl"),
        "score": eval_entry.get("score"),
        "category_scores": eval_entry.get("category_scores", {}),
    })
    reg["students"][model_id]["updated"] = now
    reg["students"][model_id]["status"] = "re-evaluated"  # xray: ignore[PY-004]

    _save_registry(reg)
    print(f"Updated eval for {model_id}: score={eval_entry.get('score')}")  # xray: ignore[PY-004]
    return 0


def cmd_gap_analysis(args: argparse.Namespace) -> int:
    """Identify weak categories for a student."""
    model_id = args.model_id
    reg = _load_registry()  # xray: ignore[PY-004]

    if model_id not in reg["students"]:
        print(f"ERROR: Student {model_id} not in registry.")  # xray: ignore[PY-004]
        return 1
  # xray: ignore-next[PY-004]
    student = reg["students"][model_id]
    if not student.get("eval_history"):
        print(f"ERROR: No eval history for {model_id}")  # xray: ignore[PY-004]
        return 1

    latest_eval = student["eval_history"][-1]
    cat_scores = latest_eval.get("category_scores", {})  # xray: ignore[PY-004]

    if not cat_scores:
        print(f"No category breakdown available for {model_id}")  # xray: ignore[PY-004]
        return 0

    avg_loss = latest_eval.get("avg_loss", 0)  # xray: ignore[PY-004]
    threshold = args.threshold  # xray: ignore[PY-004]
  # xray: ignore-next[PY-004]
    print(f"\n=== Gap Analysis: {model_id} ===")  # xray: ignore[PY-004]
    print(f"Overall avg_loss: {avg_loss:.4f}")  # xray: ignore[PY-004]
    print(f"Threshold: >{threshold:.4f}\n")  # xray: ignore[PY-004]
    print(f"{'Category':<15} {'Avg Loss':>10} {'Status'}")  # xray: ignore[PY-004]
    print("-" * 40)  # xray: ignore[PY-004]

    gaps = []
    for cat, loss in sorted(cat_scores.items()):
        if loss > threshold:
            gaps.append(cat)
            status = "GAP"  # xray: ignore[PY-004]
        else:
            status = "OK"
        print(f"{cat:<15} {loss:>10.4f} {status}")  # xray: ignore[PY-004]
  # xray: ignore-next[PY-004]
    if gaps:
        print(f"\nRemediation targets: {', '.join(gaps)}")  # xray: ignore[PY-004]
        print(f"Suggested: gather more training data for categories: {', '.join(gaps)}")  # xray: ignore[PY-004]
    else:
        print(f"\nNo gaps found above threshold {threshold}.")  # xray: ignore[PY-004]

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Side-by-side comparison of two or more students."""
    model_ids = args.model_ids
    reg = _load_registry()

    entries = []  # xray: ignore[PY-004]
    for mid in model_ids:
        if mid not in reg["students"]:
            print(f"WARNING: {mid} not in registry, skipping.")  # xray: ignore[PY-004]
            continue
        entries.append((mid, reg["students"][mid]))  # xray: ignore[PY-004]

    if len(entries) < 2:
        print("ERROR: Need at least 2 registered students to compare.")  # xray: ignore[PY-004]
        return 1

    # Header  # xray: ignore[PY-004]
    col_w = 18  # xray: ignore[PY-004]
    header = f"{'Metric':<20}" + "".join(f"{mid:<{col_w}}" for mid, _ in entries)  # xray: ignore[PY-004]
    print(f"\n{'='*len(header)}")  # xray: ignore[PY-004]
    print(header)  # xray: ignore[PY-004]
    print(f"{'='*len(header)}")  # xray: ignore[PY-004]
  # xray: ignore-next[PY-004]
    # Status
    row = f"{'Status':<20}" + "".join(f"{s.get('status', '?'):<{col_w}}" for _, s in entries)
    print(row)  # xray: ignore[PY-004]

    # SFT loss
    row = f"{'SFT Loss':<20}"
    for _, s in entries:  # xray: ignore[PY-004]
        v = s.get("loss_history", {}).get("sft_final_loss")
        row += f"{f'{v:.4f}' if v is not None else 'n/a':<{col_w}}"
    print(row)  # xray: ignore[PY-004]

    # DPO loss
    row = f"{'DPO Loss':<20}"
    for _, s in entries:  # xray: ignore[PY-004]
        v = s.get("loss_history", {}).get("dpo_final_loss")
        row += f"{f'{v:.4f}' if v is not None else 'n/a':<{col_w}}"
    print(row)  # xray: ignore[PY-004]

    # Latest score
    row = f"{'Eval Score':<20}"
    for _, s in entries:
        eh = s.get("eval_history", [])  # xray: ignore[PY-004]
        v = eh[-1].get("score") if eh else None
        row += f"{f'{v:.4f}' if v is not None else 'n/a':<{col_w}}"
    print(row)  # xray: ignore[PY-004]

    # Latest eval loss
    row = f"{'Eval Loss':<20}"
    for _, s in entries:
        eh = s.get("eval_history", [])  # xray: ignore[PY-004]
        v = eh[-1].get("avg_loss") if eh else None
        row += f"{f'{v:.4f}' if v is not None else 'n/a':<{col_w}}"
    print(row)  # xray: ignore[PY-004]

    # Category breakdown
    all_cats: set[str] = set()
    for _, s in entries:
        eh = s.get("eval_history", [])
        if eh:
            all_cats.update(eh[-1].get("category_scores", {}).keys())

    for cat in sorted(all_cats):
        row = f"  {cat:<18}"
        for _, s in entries:
            eh = s.get("eval_history", [])  # xray: ignore[PY-004]
            v = eh[-1].get("category_scores", {}).get(cat) if eh else None
            row += f"{f'{v:.4f}' if v is not None else 'n/a':<{col_w}}"
        print(row)  # xray: ignore[PY-004]

    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export registry as a Markdown report."""
    reg = _load_registry()
    students = reg.get("students", {})  # xray: ignore[PY-004]

    if not students:
        print("Registry is empty, nothing to export.")  # xray: ignore[PY-004]
        return 0

    lines = ["# Student Registry Report", ""]
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Total students: {len(students)}")
    lines.append("")

    for mid, s in sorted(students.items()):
        lines.append(f"## {mid}")
        lines.append(f"- **Status:** {s.get('status', 'unknown')}")
        lines.append(f"- **Created:** {s.get('created', 'n/a')}")
        lineage = s.get("lineage", {})
        lines.append(f"- **Base model:** {lineage.get('base_model', 'n/a')}")

        loss = s.get("loss_history", {})
        sft = loss.get("sft_final_loss")
        dpo = loss.get("dpo_final_loss")
        lines.append(f"- **SFT loss:** {f'{sft:.4f}' if sft is not None else 'n/a'}")
        lines.append(f"- **DPO loss:** {f'{dpo:.4f}' if dpo is not None else 'n/a'}")

        eh = s.get("eval_history", [])
        if eh:
            latest = eh[-1]
            lines.append(f"- **Latest eval score:** {latest.get('score')}")
            lines.append(f"- **Latest eval loss:** {latest.get('avg_loss')}")
            cats = latest.get("category_scores", {})
            if cats:
                lines.append(f"- **Categories:** {', '.join(f'{k}={v:.4f}' for k, v in cats.items())}")

        gguf = s.get("gguf_variants", [])
        if gguf:
            lines.append(f"- **GGUF exports:** {', '.join(g.get('quant', '?') for g in gguf)}")

        lines.append("")
  # xray: ignore-next[PY-004]
    output = args.output or "student_report.md"
    Path(output).write_text("\n".join(lines), encoding="utf-8")
    print(f"Exported registry report to {output}")  # xray: ignore[PY-004]
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Student Registry — longitudinal model tracking.",
        epilog="""\
examples:
  %(prog)s register --saves-tag zena007 --variant-id B
  %(prog)s list
  %(prog)s update-eval --model-id zena007/B --eval-file eval_scorecards.jsonl
  %(prog)s gap-analysis --model-id zena007/B
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # register
    p = sub.add_parser("register", help="Register a student from forge results.")
    p.add_argument("--saves-tag", required=True, help="Tag from run_student_forge.py.")
    p.add_argument("--variant-id", required=True, help="Variant ID (e.g., B).")

    # list
    sub.add_parser("list", help="List all registered students.")

    # update-eval
    p = sub.add_parser("update-eval", help="Append new eval scores.")
    p.add_argument("--model-id", required=True, help="Model ID (e.g., zena007/B).")
    p.add_argument("--eval-file", required=True, help="Path to eval_scorecards.jsonl.")

    # gap-analysis
    p = sub.add_parser("gap-analysis", help="Identify weak categories.")
    p.add_argument("--model-id", required=True, help="Model ID.")
    p.add_argument("--threshold", type=float, default=3.0, help="Loss threshold for gap detection.")

    # compare
    p = sub.add_parser("compare", help="Compare two or more students.")
    p.add_argument("--model-ids", nargs="+", required=True, help="Model IDs to compare.")

    # export
    p = sub.add_parser("export", help="Export registry as Markdown.")
    p.add_argument("--output", help="Output file path (default: student_report.md).")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1

    commands = {
        "register": cmd_register,
        "list": cmd_list,
        "update-eval": cmd_update_eval,
        "gap-analysis": cmd_gap_analysis,
        "compare": cmd_compare,
        "export": cmd_export,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
