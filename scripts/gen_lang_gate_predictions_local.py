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

"""Generate local predictions for language-gate eval from a merged local model.

Input spec rows must contain:
- id (str)
- prompt (str)

Output rows:
- id (str)
- response (str)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch  # xray: ignore[SEC-015]
from transformers import AutoModelForCausalLM, AutoTokenizer  # xray: ignore[SEC-015]


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


def _build_prompt(user_prompt: str, policy_prefix: str) -> str:
    # Keep prompt format aligned with the "default" template style used in local runs.
    if policy_prefix:
        return f"Human: {policy_prefix}\\n\\n{user_prompt}\\nAssistant:"
    return f"Human: {user_prompt}\\nAssistant:"


def _extract_response(full_text: str) -> str:
    marker = "Assistant:"
    idx = full_text.rfind(marker)
    if idx >= 0:
        return full_text[idx + len(marker) :].strip()
    return full_text.strip()


def _requests_unsupported_language(prompt: str, input_language: str) -> bool:
    allowed_input_langs = {"en", "ro", "fr", "hu", ""}
    if input_language and input_language.lower() not in allowed_input_langs:
        return True

    low = (prompt or "").lower()
    unsupported_patterns = [
        r"translate\s+to\s+spanish",
        r"responde\s+en\s+espa",
        r"bitte\s+antworte\s+auf\s+deutsch",
        r"traduci\s+in\s+italiano",
        r"\bspanish\b",
        r"\bespa[nn]ol\b",
        r"\bdeutsch\b",
        r"\bgerman\b",
        r"\bitaliano\b",
        r"\bitalian\b",
    ]
    return any(re.search(pattern, low) for pattern in unsupported_patterns)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate language-gate predictions using a local merged model.")
    parser.add_argument("--model", required=True, help="Path to merged local model directory.")
    parser.add_argument("--spec", required=True, help="Path to language-gate eval spec jsonl.")
    parser.add_argument("--out", required=True, help="Path to output predictions jsonl.")
    parser.add_argument("--max-new-tokens", type=int, default=96, help="Maximum new tokens to generate.")
    parser.add_argument(
        "--policy-prefix",
        default="",
        help="Optional policy instruction prepended to each prompt.",
    )
    parser.add_argument(
        "--guard-language-policy",
        action="store_true",
        help="If set, unsupported language requests are refused before model generation.",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    spec_path = Path(args.spec)
    out_path = Path(args.out)

    rows = _load_jsonl(spec_path)
  # xray: ignore-next[SEC-007]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model.eval()  # xray: ignore[SEC-007]

    # Use CPU-safe inference path for local reproducibility.
    device = torch.device("cpu")
    model.to(device)

    outputs: list[dict] = []
    for row in rows:
        sample_id = row.get("id", "")
        prompt = row.get("prompt", "")
        input_language = str(row.get("input_language", "") or "")
        if not sample_id or not prompt:
            continue

        if args.guard_language_policy and _requests_unsupported_language(prompt, input_language):
            outputs.append(
                {
                    "id": sample_id,
                    "response": "I can only support English, Romanian, French, and Hungarian. Please choose one of these languages.",
                }
            )
            continue

        input_text = _build_prompt(prompt, args.policy_prefix)
        inputs = tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                no_repeat_ngram_size=3,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        response = _extract_response(text)
        outputs.append({"id": sample_id, "response": response})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in outputs:  # xray: ignore[PY-004]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")  # xray: ignore[PY-004]

    print(f"predictions_saved={out_path}")  # xray: ignore[PY-004]
    print(f"total_predictions={len(outputs)}")  # xray: ignore[PY-004]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
