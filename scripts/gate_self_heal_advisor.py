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

import argparse
import json
import re
from pathlib import Path
from typing import Any

import requests  # xray: ignore[SEC-015]


def _read_log_tail(path: Path, max_chars: int = 12000) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def _rule_based_suggestions(log_tail: str) -> list[str]:
    suggestions: list[str] = []
    rules: list[tuple[str, str]] = [
        (
            r"No `target_modules` passed|target_modules",
            "LoRA target discovery failed. For GPT-2 style models ensure Conv1D modules are discovered or set lora_target explicitly.",
        ),
        (
            r"KeyError: 'instruction'|KeyError: \"instruction\"",
            "Dataset column mapping mismatch. Add prompt/chosen/rejected column mappings in data/dataset_info.json for ranking datasets.",
        ),
        (
            r"ConnectionError|ConnectTimeout|ReadTimeout|Failed to establish a new connection",
            "Endpoint connectivity issue. Verify api_base, model serving process, and retry with longer ApiWaitSeconds.",
        ),
        (
            r"CUDA out of memory|out of memory",
            "OOM detected. Reduce per_device_train_batch_size and/or increase gradient_accumulation_steps.",
        ),
        (
            r"WinError 127|_torchaudio",
            "torchaudio binary mismatch. Align torch/torchaudio versions or disable audio-specific path for text workflows.",
        ),
        (
            r"TypeError: Pickler\._batch_setitems",
            "Likely datasets/dill/multiprocess incompatibility. Upgrade to a coherent version set for Python 3.14.",
        ),
    ]

    for pattern, suggestion in rules:
        if re.search(pattern, log_tail, flags=re.IGNORECASE):
            suggestions.append(suggestion)

    if not suggestions:
        suggestions.append("No known signature matched. Inspect the step log and rerun with --EnableSelfHeal and a small helper model endpoint.")

    return suggestions


def _llm_suggestions(log_tail: str, api_base: str, model: str) -> list[str]:
    api_base = api_base.rstrip("/")
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a concise ML ops incident helper. Return JSON list of actionable, low-risk fixes.",
            },
            {
                "role": "user",
                "content": (
                    "Given this gate failure log tail, return up to 5 concrete remediation steps in JSON array format.\n\n"
                    + log_tail
                ),
            },
        ],
        "temperature": 0.0,
    }

    resp = requests.post(f"{api_base}/chat/completions", json=payload, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]

    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return [str(item) for item in parsed][:5]
    except json.JSONDecodeError:  # xray: ignore[QUAL-002]
        pass

    return [line.strip("- ").strip() for line in content.splitlines() if line.strip()][:5]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze cocktail gate failures and suggest recovery actions.")
    parser.add_argument("--step", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--api-base", default="")
    parser.add_argument("--model", default="")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"[self-heal] Log not found: {log_path}")  # xray: ignore[PY-004]
        return

    log_tail = _read_log_tail(log_path)
    print(f"[self-heal] Step: {args.step}")  # xray: ignore[PY-004]
    print(f"[self-heal] Log: {log_path}")  # xray: ignore[PY-004]

    suggestions = _rule_based_suggestions(log_tail)
    print("[self-heal] Rule-based suggestions:")  # xray: ignore[PY-004]
    for idx, item in enumerate(suggestions, 1):
        print(f"  {idx}. {item}")  # xray: ignore[PY-004]

    if args.api_base and args.model:
        try:
            llm_tips = _llm_suggestions(log_tail, args.api_base, args.model)
            print("[self-heal] LLM suggestions:")  # xray: ignore[PY-004]
            for idx, item in enumerate(llm_tips, 1):
                print(f"  {idx}. {item}")  # xray: ignore[PY-004]
        except Exception as exc:  # xray: ignore[QUAL-011]
            print(f"[self-heal] LLM suggestion call failed: {exc}")  # xray: ignore[PY-004]


if __name__ == "__main__":
    main()
