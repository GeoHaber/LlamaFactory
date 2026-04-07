# AD Coordinator Benchmark Guide

This guide provides a quick baseline benchmark for the Python review and Python-to-Rust translation workflow.

## Prerequisites

1. Start API server with your local model:

```bash
llamafactory-cli api --model_name_or_path <YOUR_MODEL_PATH> --template <YOUR_TEMPLATE>
```

2. Prepare benchmark prompts (sample provided):
- `data/python_rust_bench_demo.jsonl`

3. For WebUI-based runs, keep hardware auto-tune enabled in Chat tab to let runtime choose:
- backend (huggingface/vllm/sglang/ktransformers)
- inference dtype
- coordinator policy fallback on constrained hardware
- low-memory quantization defaults

## Run Benchmark

Fast policy:

```bash
python scripts/benchmark_ad_coordinator.py \
  --dataset data/python_rust_bench_demo.jsonl \
  --api_base http://127.0.0.1:8000/v1 \
  --policy fast \
  --use_ad_coordinator
```

Balanced policy:

```bash
python scripts/benchmark_ad_coordinator.py \
  --dataset data/python_rust_bench_demo.jsonl \
  --api_base http://127.0.0.1:8000/v1 \
  --policy balanced \
  --use_ad_coordinator
```

Quality policy:

```bash
python scripts/benchmark_ad_coordinator.py \
  --dataset data/python_rust_bench_demo.jsonl \
  --api_base http://127.0.0.1:8000/v1 \
  --policy quality \
  --use_ad_coordinator
```

Policy comparison in one run:

```bash
python scripts/benchmark_ad_coordinator.py \\
  --dataset data/python_rust_bench_demo.jsonl \\
  --api_base http://127.0.0.1:8000/v1 \\
  --compare_policies \\
  --use_ad_coordinator
```

## Outputs

Each run writes:
- JSON details: `benchmark_output/<timestamp>/coordinator_<policy>_results.json`
- Markdown summary: `benchmark_output/<timestamp>/coordinator_<policy>_summary.md`

When `--compare_policies` is used:
- `benchmark_output/<timestamp>/coordinator_compare_results.json`
- `benchmark_output/<timestamp>/coordinator_compare_summary.md`

## Baseline Metrics to Track

- p50 latency (ms)
- p95 latency (ms)
- mean latency (ms)
- pass rate over expected substrings (proxy quality)

## Next Improvements

- Replace substring proxy with compile-and-test harness for generated Rust.
- Add correctness evaluator on hidden reference tests.
- Use the WebUI backend calibration pass before benchmarking to reduce backend bias.
