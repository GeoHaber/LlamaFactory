# Distillation Starter Templates

These templates bootstrap a local distillation workflow for Python review and Python-to-Rust translation.

## Files

- `student_sft_distill.yaml`: teacher-trace supervised distillation stage.
- `student_dpo_distill.yaml`: preference optimization stage after SFT.

## Expected Datasets

Create dataset entries in `data/dataset_info.json` for:
- `distill_python_rust_train`
- `distill_python_rust_pref`

Recommended fields:
- SFT set: instruction, input, output (or chat style messages)
- DPO set: prompt, chosen, rejected

## Usage

```bash
llamafactory-cli train examples/distillation/student_sft_distill.yaml
llamafactory-cli train examples/distillation/student_dpo_distill.yaml
```

## Notes

- Start with LoRA for fast iteration and low VRAM.
- After validating quality, merge and quantize for local serving.
- Keep a small benchmark set for compile success, test pass rate, and latency.

---

## Multi-Teacher Consensus Distillation

A higher-quality distillation pipeline that uses multiple GGUF teachers concurrently
to generate responses, then splits them into consensus (SFT) and conflict (DPO) training data.

### Architecture (v4 -- SPSC Ring-Buffer FIFO Dispatch)

```
[Worker-Teacher-A] --produce--> SPSCRingBuffer[2048] --consume--> |
[Worker-Teacher-B] --produce--> SPSCRingBuffer[2048] --consume--> |- [Collector] -> checkpoints
[Worker-Teacher-N] --produce--> SPSCRingBuffer[2048] --consume--> |
```

Key properties:
- **Lock-free hot path**: SPSC ring buffers use CPython GIL-atomic integer stores -- zero mutexes on the critical path.
- **True concurrency**: llama-cpp releases the GIL during C++ matmuls, so multiple teachers run in parallel.
- **RAM-pressure throttling**: `RAMPressureThrottle` with hysteresis gates workers when system memory is tight (`--ram-pause-pct 12 --ram-resume-pct 22`).
- **Checkpoint/resume**: per-teacher JSONL checkpoints enable restartable generation; kills mid-flight lose at most one sample per teacher.
- **Adaptive budgets**: `--adaptive-budgets` automatically selects `max_tokens` and `temperature` based on prompt category (translation, code, reasoning, chat).
- **Model pre-warming**: all GGUF models are loaded into InProcessAdapter cache before workers start.

### Teacher Manifest

Create a JSON manifest describing each teacher:

```json
{
  "teachers": [
    {
      "name": "gemma-4-26B-A4B",
      "backend": "inprocess",
      "model_path": "/path/to/gemma-4-26B-A4B-Q4_K_M.gguf"
    },
    {
      "name": "Qwen2.5-14B-Instruct",
      "backend": "inprocess",
      "model_path": "/path/to/Qwen2.5-14B-Instruct-Q4_K_M.gguf"
    }
  ]
}
```

### Dispatch Modes

| Mode | Flag | Description |
|------|------|-------------|
| Sequential | `--dispatch-mode teacher-sequential` | Load one teacher at a time; lower RAM, slower |
| FIFO (default) | `--dispatch-mode teacher-fifo` | Concurrent workers with SPSC ring buffers; ~1.3x faster |

### FIFO Depth Tuning

| Flag | Behavior |
|------|----------|
| `--fifo-size 0` | **Auto mode** (recommended): provisions 2048 slots per teacher |
| `--fifo-size N` | Manual: use N slots per ring buffer |
| (omitted) | Default: 256 slots |

Since inference takes seconds per item and checkpoint writes take milliseconds,
the consumer always drains faster than producers fill. The 2K auto depth
provides ample headroom with no measurable overhead.

### End-to-End Pipeline

The full pipeline is orchestrated by `scripts/run_zena007_end_to_end.ps1`:

```
Generation -> Purify -> Config Gen -> Dataset Registration -> SFT -> DPO -> Merge -> Smoke Test
```

#### Step 1: Generate multi-teacher responses

```bash
python scripts/multi_teacher_generate.py \
  --manifest data/zena007/teacher_manifest.json \
  --prompts data/zena007_prompts.jsonl \
  --out data/zena007/teacher_responses.jsonl \
  --max-tokens 512 --temperature 0.7 \
  --adaptive-budgets \
  --dispatch-mode teacher-fifo \
  --fifo-size 0 \
  --ram-pause-pct 12 --ram-resume-pct 22
```

Output: `teacher_responses.jsonl` with attributed responses from all teachers per prompt.

#### Step 2: Purify into consensus/conflict splits

```bash
python scripts/purify_teacher_outputs.py \
  --input data/zena007/teacher_responses.jsonl \
  --out-dir data/zena007/purified \
  --answer-threshold 0.85 --reason-threshold 0.6
```

Output:
- `consensus_sft.jsonl` -- prompts where teachers agree (used for SFT)
- `conflict_dpo.jsonl` -- prompts where teachers disagree (used for DPO preference pairs)

#### Step 3: Train student

```bash
llamafactory-cli train examples/distillation/auto/zena007_sft.yaml
llamafactory-cli train examples/distillation/auto/zena007_dpo.yaml
```

#### Step 4: Merge and evaluate

```bash
llamafactory-cli export examples/distillation/auto/zena007_merge.yaml
```

### Benchmarking Dispatch Modes

Use the benchmark harness to compare sequential vs FIFO on a subset:

```bash
python scripts/benchmark_multi_teacher_dispatch.py \
  --manifest data/zena007/teacher_manifest.json \
  --prompts data/zena007_prompts.jsonl \
  --output-dir benchmark_output/multi_teacher_dispatch \
  --sample-count 2 \
  --teachers Mistral-7B-Instruct-v0.3 Qwen2.5-14B-Instruct \
  --max-tokens 96 --temperature 0.3
```
