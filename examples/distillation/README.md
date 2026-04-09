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

---

## Distillation Studio (Web UI)

For users who prefer a graphical launcher, the same pipeline is wrapped in a FastAPI
+ vanilla-HTML web UI at `scripts/distill_server.py` + `scripts/distill.html`.

**Launch:**

```bash
python scripts/distill_server.py       # or Run_me.bat on Windows
# then open http://localhost:7870
```

Requires Python >= 3.14.

### Top chrome (shared across every tab)

Two thin rows sit at the very top of the window and stay visible on every tab:

| Row               | Contents                                                                          |
|-------------------|-----------------------------------------------------------------------------------|
| **Zena topbar**   | Circular **Dean Zena avatar** (top-left, clickable → opens the settings panel), "University of Distillation" brand, and the tab nav chips (Studio · 1 Enrollment · 2 Teaching · 3 Graduation · 4 Exam). |
| **Metrics rail**  | Live system telemetry flush under the topbar: GPU % · VRAM · CPU · RAM · DISK · NET · TOK/S · ETA · bottleneck bottle. 1 Hz updates from the backend. |

There is **no hamburger icon** and **no top-bar STOP button**. Clicking the Zena avatar
slides the Settings panel in from the **left side, tucked directly under Zena**. The
only STOP control lives inside the Run Status Card (see below).

The inner tab-header strip that used to be duplicated on top of every tab pane has
been removed — the Zena topbar chips are the single source of navigation.

### Internationalization (i18n)

The Settings panel offers six languages that translate every menu, label, and tooltip
on the page: **English, French, Italian, Romanian, Hungarian, Hebrew**. The current
language is persisted in `localStorage` under `zena.lang`; Hebrew flips the document
to RTL (`html[dir="rtl"]`) and mirrors the Settings panel to slide in from the right.

### Studio-as-launcher layout

The UI has four tabs, but only one of them starts runs:

| Tab            | Role                                                                  |
|----------------|-----------------------------------------------------------------------|
| **Enrollment** | Pick Dean, Teachers, enroll Students with skills/languages            |
| **Studio**     | **Single source-of-truth launcher** — Start/Stop + all pipeline knobs |
| **Teaching**   | Read-only live monitor of stages 1-3 (generate/halluc/purify)         |
| **Graduation** | Read-only live monitor of stages 4-11 (train/merge/eval/dashboard)    |

The Studio tab uses a **Start ↔ Run Status Card slot-swap**: while idle, a big green
**Start Training** button occupies the centerpiece slot. The moment you press it,
`setPipelineRunning(true)` hides the Start button and unveils the Run Status Card in
the exact same position. The card is compact (stage label and metrics in the same
font-size as the STOP button, ~13 px) with this layout:

```
[ STOP / RESTART ]   STAGE  ·  STEP  LOSS  EPOCH  ETA  GOLD  SILVER  DROP
                     ▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░  (progress bar)
```

- The **STOP / RESTART button is on the left**, before the progress body. STOP is
  visible while the run is live; once the pipeline reaches a terminal state (done or
  fail) the button swaps to RESTART (which re-invokes `studioStart()`).
- STOP hits `POST /api/pipeline/stop`, which flips the cancel flag and kills the
  active subprocess cleanly. Partial artifacts in `saves/{tag}/` are preserved.

### Multi-student sequential batch training

When you enroll **2 or more** students on the Enrollment tab, pressing Start in Studio
trains them all sequentially in one run:

```
Stage 1-3 (shared)      Stage 4-11 (per-student loop)
┌──────────────────┐    ┌──────────────────────────────┐
│ Teacher generate │───▶│ Student 1: configs→SFT→DPO→  │
│ Halluc gates     │    │            merge→gguf→eval   │
│ Purify           │    │            →dashboard→PM     │
└──────────────────┘    │ Student 2: (same)            │
                        │ Student N: (same)            │
                        └──────────────────────────────┘
```

- **Stages 1-3 run once shared** — the purified GOLD/SILVER dataset is reused across
  every student in the batch. The curriculum prompt filter uses the **union** of all
  enrolled students' skills and languages so no student is starved of relevant prompts.
- **Stages 4-11 loop per student** with a per-student saves tag
  `{saves_tag}_{safe_student_name}` passed as `--tag` to `gen_distill_configs.py` and
  `slim_down.py` and as `--saves-tag` to `eval_student_panel.py` + `graduation_dashboard.py`.
  This guarantees unique config YAMLs, unique `saves/{s_tag}/...` adapter/merged/gguf
  directories, and unique evaluation scorecards per student.
- **STOP cancels the whole batch** — pressing STOP mid-student kills the running
  subprocess and breaks out of the loop. The pipeline reports the failed student as
  `status: "cancelled"` (not `"fail"`) in its `student_end` SSE event.
- **Single-student runs are fully back-compat** — with one enrolled student the loop
  iterates once and `s_tag = saves_tag`, matching the old flow byte-for-byte.

**Why sequential, not parallel?** Single-GPU training is compute-bound — two trainers
time-slice the same CUDA cores so wall-clock is roughly the same as sequential, often
worse due to kernel contention. Sequential also gives simpler STOP semantics, cleaner
progress tracking, and recoverable failures. On multi-GPU hosts, parallel training
(one trainer per device) would warrant a different approach.

### Studio SSE event types

The browser consumes a single SSE stream from `POST /api/pipeline/start`. Event types:

| Type              | Payload                                                   | Purpose                                    |
|-------------------|-----------------------------------------------------------|--------------------------------------------|
| `stage`           | `stage, status, msg`                                      | Per-stage lifecycle (running/done/fail/…)  |
| `log`             | `stage, text`                                             | Subprocess stdout line-by-line             |
| `train_progress`  | `stage, step, total, loss, epoch, eta`                    | Parsed trainer_log.jsonl                   |
| `metrics`         | `cpu, ram, gpu, vram, bottleneck, …`                      | Top metrics rail sampler (1 Hz)            |
| `student_begin`   | `idx, total, name, hf_id, tag`                            | Batch: new student starting stages 4-11    |
| `student_end`     | `idx, total, name, tag, status`                           | Batch: student finished (done/fail/cancelled) |
| `done_all`        | `gold, silver, drop, report, batch_size, batch_status`    | Pipeline complete                          |
| `error`           | `msg`                                                     | Fatal early-exit                           |

The Run Status Card shows a `Student N / M · name` batch cell during multi-student
runs (hidden for single-student). Between students the card's metric cells
(STEP/LOSS/EPOCH/ETA) and progress bar reset so each student starts with a fresh UI.
Teacher-generation progress (stages 1-3) also flows into the same card's progress
bar and cells so the centerpiece is always the single place to look during a run.
