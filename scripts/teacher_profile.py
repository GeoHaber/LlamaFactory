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

"""Teacher Quality Profile — per-teacher analysis of multi-teacher generation.

Reads teacher_responses.jsonl and purification outputs to produce a report
showing each teacher's agreement rate, response length, category performance,
and contribution to GOLD/SILVER/DROP tiers.

Usage:
    python scripts/teacher_profile.py \\
        --responses data/zena007/teacher_responses.jsonl \\
        --purified-dir data/zena007/purified

    python scripts/teacher_profile.py \\
        --responses data/zena007/teacher_responses.jsonl \\
        --purified-dir data/zena007/purified \\
        --json-out saves/zena007/teacher_profile.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                pass
    return rows


def _extract_category(row: dict) -> str:
    """Extract prompt category from a response row."""
    cat = row.get("prompt_category", "")
    if cat:
        return cat
    sid = row.get("id", "")
    if sid.startswith("tr-detect"):
        return "detect"
    if sid.startswith("tr-"):
        return "translation"
    if sid.startswith("chat-"):
        return "chat"
    if sid.startswith("ocr-"):
        return "ocr"
    return "other"


def build_profile(
    responses: list[dict],
    gold_rows: list[dict] | None = None,
    silver_rows: list[dict] | None = None,
    drop_rows: list[dict] | None = None,
) -> dict:
    """Build per-teacher quality profile from generation data.

    Returns a dict with teacher-level stats and an overall summary.
    """
    # Per-teacher accumulators
    teacher_stats: dict[str, dict] = defaultdict(lambda: {
        "total_responses": 0,
        "response_lengths": [],
        "categories": Counter(),
        "in_majority": 0,
        "in_minority": 0,
        "gold_source": 0,
        "silver_chosen": 0,
        "silver_rejected": 0,
    })

    # Count per-teacher contributions from responses
    for row in responses:
        teachers = row.get("teachers", {})
        category = _extract_category(row)

        for t_name, t_data in teachers.items():
            stats = teacher_stats[t_name]
            stats["total_responses"] += 1
            raw = t_data.get("raw", t_data.get("answer", ""))
            stats["response_lengths"].append(len(raw))
            stats["categories"][category] += 1

    # Cross-reference with purification outputs
    gold_sources: Counter = Counter()
    silver_chosen: Counter = Counter()
    silver_rejected: Counter = Counter()

    if gold_rows:
        for row in gold_rows:
            src = row.get("source_teacher", "")
            if src:
                gold_sources[src] += 1
            for t in row.get("agreeing_teachers", []):
                if t in teacher_stats:
                    teacher_stats[t]["in_majority"] += 1

    if silver_rows:
        for row in silver_rows:
            ct = row.get("chosen_teacher", "")
            rt = row.get("rejected_teacher", "")
            if ct:
                silver_chosen[ct] += 1
            if rt:
                silver_rejected[rt] += 1

    # Assign purification stats
    for t_name, stats in teacher_stats.items():
        stats["gold_source"] = gold_sources.get(t_name, 0)
        stats["silver_chosen"] = silver_chosen.get(t_name, 0)
        stats["silver_rejected"] = silver_rejected.get(t_name, 0)

    # Build final profile
    profiles: list[dict] = []
    for t_name in sorted(teacher_stats.keys()):
        stats = teacher_stats[t_name]
        lengths = stats["response_lengths"]
        total = stats["total_responses"]
        avg_len = sum(lengths) / len(lengths) if lengths else 0
        min_len = min(lengths) if lengths else 0
        max_len = max(lengths) if lengths else 0

        # Agreement rate: how often this teacher was in the majority
        agree_rate = stats["in_majority"] / total if total > 0 else 0.0

        profiles.append({
            "teacher": t_name,
            "total_responses": total,
            "avg_response_length": round(avg_len),
            "min_response_length": min_len,
            "max_response_length": max_len,
            "agreement_rate": round(agree_rate, 4),
            "gold_source_count": stats["gold_source"],
            "silver_chosen_count": stats["silver_chosen"],
            "silver_rejected_count": stats["silver_rejected"],
            "categories": dict(stats["categories"].most_common()),
        })

    return {
        "teacher_count": len(profiles),
        "total_prompts": len(responses),
        "teachers": profiles,
    }


def print_profile(profile: dict) -> None:
    """Print a human-readable teacher profile report."""
    print("\n=== Teacher Quality Profile ===")  # xray: ignore[PY-004]
    print(f"Teachers: {profile['teacher_count']}  |  Prompts: {profile['total_prompts']}")  # xray: ignore[PY-004]

    for t in profile["teachers"]:
        name = t["teacher"]
        print(f"\n  {name}")  # xray: ignore[PY-004]
        print(f"    Responses:     {t['total_responses']}")  # xray: ignore[PY-004]
        print(f"    Avg length:    {t['avg_response_length']} chars")  # xray: ignore[PY-004]
        print(f"    Agreement:     {t['agreement_rate']:.0%}")  # xray: ignore[PY-004]
        print(f"    GOLD source:   {t['gold_source_count']}")  # xray: ignore[PY-004]
        print(f"    SILVER chosen: {t['silver_chosen_count']}")  # xray: ignore[PY-004]
        print(f"    SILVER reject: {t['silver_rejected_count']}")  # xray: ignore[PY-004]
        cats = t["categories"]
        if cats:
            cat_str = ", ".join(f"{c}={n}" for c, n in cats.items())
            print(f"    Categories:    {cat_str}")  # xray: ignore[PY-004]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Teacher quality profile report from multi-teacher generation.",
        epilog="""\
examples:
  %(prog)s --responses data/zena007/teacher_responses.jsonl
  %(prog)s --responses data/zena007/teacher_responses.jsonl --purified-dir data/zena007/purified
  %(prog)s --responses data/zena007/teacher_responses.jsonl --json-out profile.json
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--responses", required=True, help="Path to teacher_responses.jsonl.")
    parser.add_argument("--purified-dir", help="Path to purified output directory (optional, enriches report).")
    parser.add_argument("--json-out", help="Write profile report as JSON to this path.")
    args = parser.parse_args()

    resp_path = Path(args.responses)
    if not resp_path.exists():
        print(f"ERROR: {resp_path} not found", file=sys.stderr)
        return 1

    responses = _load_jsonl(resp_path)
    print(f"Loaded {len(responses)} response rows.")  # xray: ignore[PY-004]

    gold_rows = silver_rows = drop_rows = None
    if args.purified_dir:
        pdir = Path(args.purified_dir)
        gold_rows = _load_jsonl(pdir / "consensus_sft.jsonl")
        silver_rows = _load_jsonl(pdir / "conflict_dpo.jsonl")
        drop_rows = _load_jsonl(pdir / "dropped_log.jsonl")

    profile = build_profile(responses, gold_rows, silver_rows, drop_rows)
    print_profile(profile)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(profile, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nJSON report saved to {out_path}")  # xray: ignore[PY-004]

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
