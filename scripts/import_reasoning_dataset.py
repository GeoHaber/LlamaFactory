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

"""Import a pre-distilled reasoning dataset from Hugging Face into the
purification output format that the Distillation pipeline expects.

Why this exists
---------------
The standard pipeline (multi_teacher_generate -> purify) is the slow path:
spin up 1-N local teachers, generate 10-100k responses, run the 5-gate
hallucination filter, then majority-vote into GOLD/SILVER/DROP buckets.
That can take HOURS even on a strong rig.

The fast path (this script) skips all of that by treating an upstream HF
reasoning dataset (already distilled from a frontier closed-weights model
like Claude 4.6 Opus) as a pre-purified GOLD set. You go from "I have nothing"
to "I'm training" in <60 seconds for the small datasets.

After running this script, the output directory looks identical to what the
slow path would have produced:

    <output_dir>/
        consensus_sft.jsonl       <- the GOLD samples in {instruction, output}
        purification_report.json  <- synthetic: gold=N, silver=0, drop=0
        teacher_responses.jsonl   <- placeholder so existence checks pass
        upstream_meta.json        <- provenance: which HF dataset, how many rows

From there, the rest of the pipeline (gen_distill_configs -> SFT -> DPO ->
merge -> GGUF -> eval) runs exactly as it would for a slow-path run.

Supported datasets
------------------
Four well-known reasoning datasets ship with explicit converters because each
one uses a different schema. A generic "messages" fallback handles most other
shareGPT/ChatML-style datasets.

  nohurry/Opus-4.6-Reasoning-3000x-filtered  (problem/thinking/solution)
  Roman1111111/claude-opus-4.6-10000x        (messages format)
  TeichAI/claude-4.5-opus-high-reasoning-250x (messages format)
  Jackrong/Qwen3.5-reasoning-700x            (input/output, output has <think>)

Usage
-----
    python scripts/import_reasoning_dataset.py \
        --dataset Roman1111111/claude-opus-4.6-10000x \
        --output-dir data/upstream_opus46 \
        --max-rows 10000 \
        --tier GOLD
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Known datasets -- explicit per-dataset converters
# ---------------------------------------------------------------------------
# Each entry knows its filename inside the HF repo and how to convert one row
# of that schema into the canonical {instruction, output} pair we need.

THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def _strip(s: object) -> str:
    return (str(s) if s is not None else "").strip()


def _wrap_think(thinking: str, solution: str) -> str:
    """Build a `<think>...</think>\\n{solution}` string from separate fields."""
    thinking = _strip(thinking)
    solution = _strip(solution)
    if not thinking and not solution:
        return ""
    if not thinking:
        return solution
    if not solution:
        return f"<think>\n{thinking}\n</think>"
    return f"<think>\n{thinking}\n</think>\n{solution}"


def _parse_messages_field(raw: object) -> list[dict] | None:
    """Some upstream datasets store ``messages`` as a Python-repr string.

    HF datasets emitted via pandas or naive serialization often produce a
    string like ``"[{'role': 'user', ...}]"`` rather than a real list. We try
    JSON first, then literal_eval as a fallback.
    """
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            try:
                return ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                return None
    return None


def _convert_problem_thinking_solution(row: dict) -> dict | None:
    """Schema: nohurry/Opus-4.6-Reasoning-3000x-filtered."""
    problem = _strip(row.get("problem"))
    if not problem:
        return None
    output = _wrap_think(row.get("thinking", ""), row.get("solution", ""))
    if not output:
        return None
    return {
        "instruction": problem,
        "output": output,
        "category": _strip(row.get("category")) or "general",
        "difficulty": _strip(row.get("difficulty")) or "medium",
    }


def _convert_messages(row: dict) -> dict | None:
    """Schema: Roman1111111/claude-opus-4.6-10000x and TeichAI shareGPT-style."""
    msgs = _parse_messages_field(row.get("messages"))
    if not msgs:
        return None

    # Find the last user turn and the last assistant turn that follows it.
    # Simple sequential walk -- handles multi-turn dialogues but flattens to
    # a single instruction/output pair (which is what consensus_sft.jsonl
    # wants for SFT).
    user_idx = None
    asst_idx = None
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or m.get("from") or "").lower()
        if role in ("user", "human"):
            user_idx = i
        elif role in ("assistant", "gpt", "model") and user_idx is not None:
            asst_idx = i

    if user_idx is None or asst_idx is None:
        return None

    user_content = _strip(
        msgs[user_idx].get("content")
        or msgs[user_idx].get("value")
    )
    asst_content = _strip(
        msgs[asst_idx].get("content")
        or msgs[asst_idx].get("value")
    )
    if not user_content or not asst_content:
        return None

    return {
        "instruction": user_content,
        "output": asst_content,
    }


def _convert_input_output(row: dict) -> dict | None:
    """Schema: Jackrong/Qwen3.5-reasoning-700x (input is prompt, output has <think>)."""
    instr = _strip(row.get("input"))
    out = _strip(row.get("output"))
    if not instr or not out:
        return None
    return {
        "instruction": instr,
        "output": out,
        "domain": _strip(row.get("domain")) or "general",
    }


# Filename + converter for each well-known dataset.
KNOWN: dict[str, dict] = {
    "nohurry/Opus-4.6-Reasoning-3000x-filtered": {
        "filename": "distilled_corpus_400k_with_cot-filtered.jsonl",
        "convert": _convert_problem_thinking_solution,
        "label": "Opus-4.6 reasoning (filtered, separate thinking/solution)",
    },
    "Roman1111111/claude-opus-4.6-10000x": {
        "filename": "opus46_final.jsonl",
        "convert": _convert_messages,
        "label": "Opus-4.6 messages (10k, largest)",
    },
    "TeichAI/claude-4.5-opus-high-reasoning-250x": {
        "filename": "claude-opus-4.5-250x.jsonl",
        "convert": _convert_messages,
        "label": "Opus-4.5 high-reasoning (small but excellent)",
    },
    "Jackrong/Qwen3.5-reasoning-700x": {
        "filename": "distilled_stage2.jsonl",
        "convert": _convert_input_output,
        "label": "Qwen-rephrased reasoning (700)",
    },
}


# ---------------------------------------------------------------------------
# Generic auto-detection fallback
# ---------------------------------------------------------------------------

def _auto_convert(row: dict) -> dict | None:
    """Best-effort schema-agnostic converter.

    Walks the row's keys looking for {prompt, response} / {input, output} /
    {messages} / {problem, thinking, solution}-shaped data. Returns None if
    nothing matches so the caller can drop the row.
    """
    if "messages" in row:
        return _convert_messages(row)
    if "problem" in row and ("thinking" in row or "solution" in row):
        return _convert_problem_thinking_solution(row)
    if "input" in row and "output" in row:
        return _convert_input_output(row)
    if "instruction" in row and "output" in row:
        # Already in our format!
        return {
            "instruction": _strip(row["instruction"]),
            "output": _strip(row["output"]),
        }
    if "prompt" in row and "response" in row:
        return {
            "instruction": _strip(row["prompt"]),
            "output": _strip(row["response"]),
        }
    return None


# ---------------------------------------------------------------------------
# Download + convert
# ---------------------------------------------------------------------------

def _download_dataset_file(repo_id: str, filename: str, cache_dir: Path) -> Path:
    """Download a single file from a HF dataset repo."""
    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
    except ImportError as e:
        raise SystemExit(
            "huggingface_hub is required: pip install huggingface_hub"
        ) from e

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=str(cache_dir),
    )
    return Path(path)


def _resolve_dataset(repo_id: str) -> tuple[str, callable, str]:
    """Look up the canonical filename and converter for a dataset.

    For datasets not in KNOWN, we try the generic auto-detection path with the
    first .jsonl file we can find via the HF API.
    """
    if repo_id in KNOWN:
        spec = KNOWN[repo_id]
        return spec["filename"], spec["convert"], spec["label"]

    # Generic path: discover the first .jsonl in the repo.
    try:
        from huggingface_hub import HfApi  # noqa: PLC0415
    except ImportError as e:
        raise SystemExit(
            "huggingface_hub is required: pip install huggingface_hub"
        ) from e
    api = HfApi()
    info = api.dataset_info(repo_id)
    candidates = [
        s.rfilename for s in (info.siblings or [])
        if s.rfilename.endswith(".jsonl") or s.rfilename.endswith(".json")
    ]
    if not candidates:
        raise SystemExit(
            f"No .jsonl/.json file found in dataset {repo_id}. "
            "Use --filename to override."
        )
    return candidates[0], _auto_convert, "auto-detected"


def _iter_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import a HF reasoning dataset as a pre-purified GOLD set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
known datasets:
  nohurry/Opus-4.6-Reasoning-3000x-filtered    (~2.3k filtered traces)
  Roman1111111/claude-opus-4.6-10000x          (~9.6k traces, largest)
  TeichAI/claude-4.5-opus-high-reasoning-250x  (~250 high quality)
  Jackrong/Qwen3.5-reasoning-700x              (~700 Qwen-rephrased)

example:
  %(prog)s --dataset Roman1111111/claude-opus-4.6-10000x \\
           --output-dir data/upstream_opus46
""",
    )
    parser.add_argument("--dataset", required=True, help="HF dataset id, e.g. Roman1111111/claude-opus-4.6-10000x")
    parser.add_argument("--output-dir", required=True, help="Where to write consensus_sft.jsonl + sentinel files.")
    parser.add_argument("--max-rows", type=int, default=0, help="Hard cap on rows imported (0 = unlimited).")
    parser.add_argument("--tier", choices=["GOLD", "SILVER"], default="GOLD",
                        help="Tier label written into each row (default GOLD).")
    parser.add_argument("--filename", default="", help="Override the file to download from the repo.")
    parser.add_argument("--cache-dir", default="", help="Where to cache the raw HF download (default: <output-dir>/.hf_cache).")
    parser.add_argument("--no-sentinel", action="store_true",
                        help="Do not write the synthetic teacher_responses.jsonl + purification_report.json files.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else (out_dir / ".hf_cache")

    # Resolve filename + converter for the dataset
    if args.filename:
        filename = args.filename
        convert = _auto_convert
        label = "manual filename"
    else:
        filename, convert, label = _resolve_dataset(args.dataset)

    print(f"[import] dataset:  {args.dataset}")        # noqa: T201
    print(f"[import] file:     {filename}")             # noqa: T201
    print(f"[import] schema:   {label}")                # noqa: T201
    print(f"[import] target:   {out_dir / 'consensus_sft.jsonl'}")  # noqa: T201

    # Download
    print(f"[import] downloading from huggingface.co/datasets/{args.dataset}")  # noqa: T201
    src_path = _download_dataset_file(args.dataset, filename, cache_dir)
    src_size_mb = src_path.stat().st_size / 1024 / 1024
    print(f"[import] downloaded {src_size_mb:.1f} MB -> {src_path}")  # noqa: T201

    # Convert + write
    out_jsonl = out_dir / "consensus_sft.jsonl"
    n_in = n_out = n_skip = 0
    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for raw_row in _iter_jsonl(src_path):
            n_in += 1
            converted = convert(raw_row)
            if converted is None:
                n_skip += 1
                continue
            converted.setdefault("id", f"{args.dataset.replace('/', '_')}_{n_out:06d}")
            converted["source_teacher"] = args.dataset
            converted["agreeing_teachers"] = [args.dataset]
            converted["tier"] = args.tier
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            n_out += 1
            if args.max_rows and n_out >= args.max_rows:
                break

    print(f"[import] read {n_in} rows  ->  kept {n_out}  /  skipped {n_skip}")  # noqa: T201

    if n_out == 0:
        print("[import] !! no rows survived conversion. Aborting.", file=sys.stderr)  # noqa: T201
        return 2

    # Sentinel files so pipeline_start sees the same shape as a slow-path run
    if not args.no_sentinel:
        # purification_report.json -- "everything is GOLD"
        report = {
            "source": "upstream_dataset",
            "dataset": args.dataset,
            "filename": filename,
            "imported_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "gold_count": n_out,
            "silver_count": 0,
            "dropped_count": 0,
            "input_rows": n_in,
            "skipped_rows": n_skip,
        }
        (out_dir / "purification_report.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        # teacher_responses.jsonl -- empty placeholder so the
        # _responses_ok() check in pipeline_start short-circuits to "skip".
        # The file is intentionally tiny -- just enough to satisfy the
        # is_file() probe; the real content lives in consensus_sft.jsonl.
        placeholder = out_dir / "teacher_responses.jsonl"
        if not placeholder.exists():
            placeholder.write_text(
                json.dumps({
                    "_upstream_placeholder": True,
                    "dataset": args.dataset,
                    "rows_in_consensus_sft": n_out,
                }) + "\n",
                encoding="utf-8",
            )

        # upstream_meta.json -- provenance for postmortem / dashboards
        (out_dir / "upstream_meta.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        print("[import] wrote sentinel files: purification_report.json, "  # noqa: T201
              "teacher_responses.jsonl, upstream_meta.json")

    print(f"[import] DONE  ({n_out} GOLD samples ready for SFT)")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
