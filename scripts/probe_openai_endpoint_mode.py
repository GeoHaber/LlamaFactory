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
from typing import Any

import requests  # xray: ignore[SEC-015]


def _try_chat(api_base: str, model: str, timeout: int, api_key: str | None) -> tuple[bool, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "stream": False,
    }
    response = requests.post(f"{api_base.rstrip('/')}/chat/completions", json=payload, headers=headers, timeout=timeout)
    return response.status_code < 400, f"chat_status={response.status_code}"


def _try_completions(api_base: str, model: str, timeout: int, api_key: str | None) -> tuple[bool, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: dict[str, Any] = {
        "model": model,
        "prompt": "ping",
        "max_tokens": 1,
        "stream": False,
    }
    response = requests.post(f"{api_base.rstrip('/')}/completions", json=payload, headers=headers, timeout=timeout)
    return response.status_code < 400, f"completions_status={response.status_code}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe OpenAI-compatible endpoint mode support.")
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=int, default=10)
    args = parser.parse_args()

    mode = None
    errors: list[str] = []

    try:
        ok, note = _try_chat(args.api_base, args.model, args.timeout, args.api_key)
        errors.append(note)
        if ok:
            mode = "chat"
    except Exception as exc:  # noqa: BLE001  # xray: ignore[QUAL-011]
        errors.append(f"chat_probe_error={exc}")

    if mode is None:
        try:
            ok, note = _try_completions(args.api_base, args.model, args.timeout, args.api_key)
            errors.append(note)
            if ok:
                mode = "completions"
        except Exception as exc:  # noqa: BLE001  # xray: ignore[QUAL-011]
            errors.append(f"completions_probe_error={exc}")

    payload: dict[str, Any] = {"mode": mode, "errors": errors}
    print(json.dumps(payload, ensure_ascii=True))  # xray: ignore[PY-004]

    raise SystemExit(0 if mode else 1)


if __name__ == "__main__":
    main()
