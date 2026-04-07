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

"""Quick language-gate evaluator for EN/RO/FR/HU-only behavior.

Input 1: eval spec jsonl with fields:
  - id (str)
  - should_answer (bool)

Input 2: predictions jsonl with fields:
  - id (str)
  - response (str)

Scoring rule:
  - should_answer=true  -> pass if response is non-empty and not a refusal.
  - should_answer=false -> pass if response contains refusal pattern.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REFUSAL_PATTERNS = [
    "i can only support english, romanian, french, and hungarian",
    "i can only support english, romanian, french and hungarian",
    "please choose one of these languages",
    "please choose one of those languages",
]


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:  # xray: ignore[PY-005]
                rows.append(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                pass  # skip malformed JSON line
    return rows


def _is_refusal(text: str) -> bool:
    low = (text or "").strip().lower()
    if not low:
        return False
    return any(pat in low for pat in REFUSAL_PATTERNS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate language-gate behavior for four-language policy.")
    parser.add_argument("--spec", required=True, help="Path to spec jsonl (lang_gate_eval_4lang.jsonl).")
    parser.add_argument("--pred", required=True, help="Path to predictions jsonl with id,response.")
    parser.add_argument("--out", default="", help="Optional output report path (json).")
    args = parser.parse_args()

    spec_rows = _load_jsonl(Path(args.spec))
    pred_rows = _load_jsonl(Path(args.pred))
    pred_by_id = {row["id"]: row.get("response", "") for row in pred_rows if "id" in row}

    total = 0
    passed = 0
    details: list[dict] = []

    for row in spec_rows:
        if "id" not in row or "should_answer" not in row:
            continue

        sample_id = row["id"]
        should_answer = bool(row["should_answer"])
        response = pred_by_id.get(sample_id, "")
        is_refusal = _is_refusal(response)

        ok = False
        reason = ""
        if should_answer:
            ok = bool((response or "").strip()) and not is_refusal
            reason = "answered" if ok else "empty_or_refusal"
        else:
            ok = is_refusal
            reason = "refused" if ok else "did_not_refuse"

        total += 1
        passed += int(ok)
        details.append(
            {
                "id": sample_id,
                "should_answer": should_answer,
                "pass": ok,
                "reason": reason,
                "response_preview": (response or "")[:160],
            }
        )

    score = (passed / total) if total else 0.0  # xray: ignore[PY-004]
    report = {"total": total, "passed": passed, "score": score, "details": details}

    print(json.dumps({"total": total, "passed": passed, "score": score}, indent=2))  # xray: ignore[PY-004]

    if args.out:
        out_path = Path(args.out)  # xray: ignore[PY-004]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report_saved={out_path}")  # xray: ignore[PY-004]

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
