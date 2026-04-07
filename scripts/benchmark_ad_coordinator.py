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
import os
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import requests  # xray: ignore[SEC-015]


@dataclass
class BenchmarkItem:
    item_id: str
    prompt: str
    expected_substrings: list[str]


def _load_items(dataset_path: str, max_samples: int | None = None) -> list[BenchmarkItem]:
    items: list[BenchmarkItem] = []
    with open(dataset_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue

            raw = json.loads(line)  # xray: ignore[PY-005]
            item = BenchmarkItem(
                item_id=str(raw.get("id", idx)),
                prompt=raw["prompt"],
                expected_substrings=list(raw.get("expected_substrings", [])),
            )
            items.append(item)
            if max_samples is not None and len(items) >= max_samples:
                break

    return items


def _call_chat_completion(
    api_base: str,
    model: str,
    prompt: str,
    timeout: int,
    api_key: str | None,
    use_ad_coordinator: bool,
    ad_policy: str,
    ad_complexity_threshold: int,
) -> tuple[str, float]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "use_ad_coordinator": use_ad_coordinator,
        "ad_coordinator_policy": ad_policy,
        "ad_complexity_threshold": ad_complexity_threshold,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    start = time.perf_counter()
    response = requests.post(api_base.rstrip("/") + "/chat/completions", json=payload, headers=headers, timeout=timeout)  # xray: ignore[SEC-005]
    latency_ms = (time.perf_counter() - start) * 1000.0
    response.raise_for_status()

    body = response.json()
    text = body["choices"][0]["message"].get("content", "")
    return text, latency_ms


def _call_completion(
    api_base: str,
    model: str,
    prompt: str,
    timeout: int,
    api_key: str | None,
) -> tuple[str, float]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    start = time.perf_counter()
    response = requests.post(api_base.rstrip("/") + "/completions", json=payload, headers=headers, timeout=timeout)  # xray: ignore[SEC-005]
    latency_ms = (time.perf_counter() - start) * 1000.0
    response.raise_for_status()

    body = response.json()
    text = body["choices"][0].get("text", "")
    return text, latency_ms


def _probe_endpoint_mode(api_base: str, model: str, timeout: int, api_key: str | None) -> str:
    try:
        _call_chat_completion(
            api_base=api_base,
            model=model,
            prompt="ping",
            timeout=timeout,
            api_key=api_key,
            use_ad_coordinator=False,
            ad_policy="balanced",
            ad_complexity_threshold=400,
        )
        return "chat"
    except Exception:  # xray: ignore[QUAL-002, QUAL-011]
        pass

    _call_completion(
        api_base=api_base,
        model=model,
        prompt="ping",
        timeout=timeout,
        api_key=api_key,
    )
    return "completions"


def _score_output(output: str, expected_substrings: list[str]) -> dict[str, Any]:
    if not expected_substrings:
        return {"pass": None, "matched": 0, "total": 0}

    matched = sum(1 for item in expected_substrings if item in output)
    return {
        "pass": matched == len(expected_substrings),
        "matched": matched,
        "total": len(expected_substrings),
    }


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [entry["latency_ms"] for entry in results]
    pass_flags = [entry["score"]["pass"] for entry in results if entry["score"]["pass"] is not None]

    summary: dict[str, Any] = {
        "samples": len(results),
        "latency_ms": {
            "p50": statistics.median(latencies) if latencies else None,
            "p95": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies, default=None),
            "mean": statistics.fmean(latencies) if latencies else None,
        },
    }

    if pass_flags:
        passed = sum(1 for flag in pass_flags if flag)
        summary["pass_rate"] = passed / len(pass_flags)
    else:
        summary["pass_rate"] = None

    return summary


def _write_report(
    output_dir: str,
    policy: str,
    use_ad_coordinator: bool,
    summary: dict[str, Any],
    details: list[dict[str, Any]],
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    out_json = os.path.join(output_dir, f"coordinator_{policy}_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": details}, f, indent=2, ensure_ascii=False)

    out_md = os.path.join(output_dir, f"coordinator_{policy}_summary.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# AD Coordinator Benchmark Summary\n\n")
        f.write(f"- coordinator_enabled: {use_ad_coordinator}\n")
        f.write(f"- policy: {policy}\n")
        f.write(f"- samples: {summary['samples']}\n")
        f.write(f"- latency_p50_ms: {summary['latency_ms']['p50']}\n")
        f.write(f"- latency_p95_ms: {summary['latency_ms']['p95']}\n")
        f.write(f"- latency_mean_ms: {summary['latency_ms']['mean']}\n")
        f.write(f"- pass_rate: {summary['pass_rate']}\n")

def _write_compare_report(output_dir: str, compare: dict[str, dict[str, Any]]) -> None:
    out_json = os.path.join(output_dir, "coordinator_compare_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(compare, f, indent=2, ensure_ascii=False)

    out_md = os.path.join(output_dir, "coordinator_compare_summary.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# AD Coordinator Policy Comparison\n\n")
        for policy, payload in compare.items():
            summary = payload["summary"]
            f.write(f"## {policy}\n")
            f.write(f"- samples: {summary['samples']}\n")
            f.write(f"- latency_p50_ms: {summary['latency_ms']['p50']}\n")
            f.write(f"- latency_p95_ms: {summary['latency_ms']['p95']}\n")
            f.write(f"- latency_mean_ms: {summary['latency_ms']['mean']}\n")
            f.write(f"- pass_rate: {summary['pass_rate']}\n\n")


def run_benchmark(args: argparse.Namespace) -> None:
    items = _load_items(args.dataset, args.max_samples)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)

    endpoint_mode = args.endpoint_mode
    if endpoint_mode == "auto":
        endpoint_mode = _probe_endpoint_mode(args.api_base, args.model, args.timeout, args.api_key)

    policies = [args.policy]
    if args.compare_policies:
        policies = ["fast", "balanced", "quality"]

    compare_payload: dict[str, dict[str, Any]] = {}
    for policy in policies:
        details: list[dict[str, Any]] = []
        for item in items:
            if endpoint_mode == "chat":
                output, latency_ms = _call_chat_completion(
                    api_base=args.api_base,
                    model=args.model,
                    prompt=item.prompt,
                    timeout=args.timeout,
                    api_key=args.api_key,
                    use_ad_coordinator=args.use_ad_coordinator,
                    ad_policy=policy,
                    ad_complexity_threshold=args.ad_complexity_threshold,
                )
            else:
                output, latency_ms = _call_completion(
                    api_base=args.api_base,
                    model=args.model,
                    prompt=item.prompt,
                    timeout=args.timeout,
                    api_key=args.api_key,
                )
            score = _score_output(output, item.expected_substrings)
            details.append(
                {
                    "id": item.item_id,
                    "prompt": item.prompt,
                    "output": output,
                    "latency_ms": latency_ms,
                    "score": score,
                }
            )

        summary = _summarize(details)
        _write_report(output_dir, policy, args.use_ad_coordinator, summary, details)
        compare_payload[policy] = {"summary": summary}

    if args.compare_policies:
        _write_compare_report(output_dir, compare_payload)

    print(json.dumps({"output_dir": output_dir, "endpoint_mode": endpoint_mode, "policies": compare_payload}, indent=2))  # xray: ignore[PY-004]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark AD coordinator via OpenAI-compatible API.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to JSONL dataset with prompt fields.")
    parser.add_argument("--api_base", type=str, default="http://127.0.0.1:8000/v1", help="OpenAI-style API base URL.")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="Model name used by API server.")
    parser.add_argument("--api_key", type=str, default=None, help="Optional bearer token for API auth.")
    parser.add_argument("--policy", type=str, choices=["fast", "balanced", "quality"], default="balanced")
    parser.add_argument("--compare_policies", action="store_true", help="Run fast/balanced/quality and emit combined report.")
    parser.add_argument("--use_ad_coordinator", action="store_true", help="Enable AD coordinator on request.")
    parser.add_argument("--ad_complexity_threshold", type=int, default=400)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="benchmark_output")
    parser.add_argument(
        "--endpoint_mode",
        type=str,
        choices=["auto", "chat", "completions"],
        default="auto",
        help="API route mode for inference calls.",
    )
    return parser


if __name__ == "__main__":
    run_benchmark(build_parser().parse_args())
