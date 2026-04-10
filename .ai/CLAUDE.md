# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Code style (auto-fix)
make style

# Code quality check (no modifications)
make quality

# Run all tests
make test

# Run a single test file
WANDB_DISABLED=true pytest -vv --import-mode=importlib tests/path/to/test_file.py

# Run tests matching a pattern
WANDB_DISABLED=true pytest -vv --import-mode=importlib tests/ -k "test_name"

# License header check
make license

# Build package
make build
```

The project uses `uv` as the preferred package manager. Commands automatically use `uv run` / `uvx` if `uv` is available.

## Architecture

LlamaFactory has two parallel architectures controlled by the `USE_V1` environment variable:

- **v0 (default):** `api, webui > chat, eval, train > data, model > hparams > extras`
- **v1 (experimental, `USE_V1=1`):** `trainers > core > accelerator, plugins, config > utils`

Most active development happens in v0. The v1 architecture lives in `src/llamafactory/v1/`.

### Entry Points

CLI entry point is `llamafactory-cli` / `lmf` → `src/llamafactory/cli.py:main()`, which dispatches to `launcher.py` based on `USE_V1`.

Available subcommands: `train`, `chat`, `api`, `export`, `webchat`, `webui`, `env`, `version`, `help`.

### Training Flow (v0)

```
run_exp() [tuner.py]
  → read_args() → parse YAML/JSON config
  → get_train_args() → produces typed argument dataclasses
  → routes to: run_sft / run_dpo / run_ppo / run_rm / run_pt / run_kto
  → optional: export_model()
```

Training is invoked with a YAML config: `llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml`

### Configuration System

All training parameters are YAML/JSON config files. Argument parsing in `src/llamafactory/hparams/parser.py` produces four typed dataclasses:
- `ModelArguments` — model/tokenizer selection, quantization
- `DataArguments` — datasets, templates, preprocessing
- `FinetuningArguments` — LoRA rank/target, training method (sft/dpo/ppo/rm/pt/kto)
- `TrainingArguments` — extends HuggingFace's `TrainingArguments`

### Key Modules

| Module | Purpose |
|--------|---------|
| `src/llamafactory/model/loader.py` | Loads model + tokenizer; applies quantization, LoRA, patches |
| `src/llamafactory/model/patcher.py` | Model-specific compatibility patches |
| `src/llamafactory/data/template.py` | Prompt templates; `TEMPLATES` dict maps model family → format |
| `src/llamafactory/data/mm_plugin.py` | Multi-modal (image/video/audio) data handling |
| `src/llamafactory/data/processor/` | Per-stage data processors (supervised, pairwise, pretrain, etc.) |
| `src/llamafactory/train/sft/` | SFT trainer; other stages follow same structure |
| `src/llamafactory/chat/` | Inference engines: `hf_engine`, `vllm_engine`, `sglang_engine`, `kt_engine` |
| `src/llamafactory/extras/constants.py` | Enums and constants used across the project |

### Adding Support for a New Model

1. Add a prompt template to `src/llamafactory/data/template.py` in the `TEMPLATES` dict
2. Add any necessary model patches in `src/llamafactory/model/patcher.py`
3. Add multi-modal support in `src/llamafactory/data/mm_plugin.py` if needed

### Distributed Training

Multi-GPU automatically uses `torchrun`. Additional backends:
- **Ray:** Optional Ray cluster support
- **HyperParallel FSDP2:** `src/llamafactory/train/hyper_parallel/`
- **Megatron-core:** `src/llamafactory/train/mca/`

### Testing

- `tests/` — v0 tests; `tests_v1/` — v1 tests
- Most training tests require GPU hardware
- pytest markers: `@pytest.mark.slow`, `@pytest.mark.runs_on(['cuda'])`
- Always set `WANDB_DISABLED=true` when running tests

### Code Style

- Ruff for linting and formatting (line length 119, Google-style docstrings)
- Python 3.11+ syntax
- Double quotes for strings
- All new files must include Apache 2.0 license header (checked by `make license`)

### Distillation Web UI (`scripts/distill_server.py` + `scripts/distill.html`)

Standalone FastAPI app (not Gradio) on port 7870. Requires Python >= 3.14. Canonical
launcher is `Run_me.bat`, which starts `distill_server.py`.

**Run:** `python scripts/distill_server.py`

**Studio-as-launcher redesign** — the Studio tab is the single source-of-truth for
starting a run. It has **one** Start button (the monitor tabs are read-only live views;
do not add Start buttons back to them). A Roster card at the top of Studio shows enrolled
Dean/Teachers/Students from the Enrollment tab. The student picker dropdown selects which
enrolled student drives the run (not a manual HF-id text field). A STOP button in the
top bar only appears while a run is active and hits `/api/pipeline/stop`, which flips the
cancel flag and kills the registered subprocess.

Tab roles:
1. **Enrollment** — roster: pick Dean, Teachers, enroll Students with skills/languages
2. **Studio** (launcher) — Roster card, student picker, Start/Stop, all pipeline knobs
3. **Teaching** / **Graduation** — read-only live monitors (idle-hint when nothing is running)

**Sequential multi-student batch training** — when 2+ students are enrolled, one Start
trains them all sequentially:
- Stages 1-3 (teacher generate → halluc gates → purify) run **once shared**, using the
  **union** of skills/languages across all enrolled students for the curriculum prompt filter
- Stages 4-11 (configs → SFT → DPO → merge → gguf → eval → dashboard → postmortem) loop
  per student with a per-student saves tag `{user_tag}_{safe_student_name}` passed as
  `--tag` to `gen_distill_configs.py` + `slim_down.py` and `--saves-tag` to the eval/dashboard
- Per-student `saves/{s_tag}/curriculum.json` sidecar is written for each student
- STOP between students breaks the whole batch cleanly; mid-stage kill detects cancel
  via `_is_cancel_requested()` and reports status `"cancelled"` instead of `"fail"`
- Rationale: single-GPU training is compute-bound, so parallel trainers time-slice the
  same CUDA cores — sequential is ≈ same wall-clock but has simpler STOP, cleaner
  progress tracking, and recoverable failures. Multi-GPU would warrant a different approach.
- Single-student runs are fully back-compat (loop iterates once with `s_tag = tag`)

**Curriculum filtering** (`_filter_prompts_by_curriculum()`): before generation, prompts JSONL
is filtered to keep only prompts matching the curriculum. Prompt IDs encode category
(`tr-en2es-0013` = translation, `chat-en-0524` = conversation, `ocr-0001` = OCR).
Filtered file → `curriculum_prompts.jsonl`; metadata → `saves/<tag>/curriculum.json`.

**Key data structures (JS):**
- `SKILLS[]` — 8 skills with id, icon, name, complexity
- `BRAIN_TIERS[]` — 3 quality tiers (OK/Good/Excellent) → HF model IDs
- `_selectedSkills`, `_selectedLangs`, `_selectedCodeLangs` — Sets of enrolled choices
- `_availableTiers[]` — computed by `recalcTiers()` based on skill complexity
- `window._students` — array of enrolled student objects `{ name, hf_id, quality, skills, languages, code_langs, ... }`
- `_studioStudentIdx` — index of the student currently selected in the picker dropdown
- `STAGE_TO_TAB` — maps stage ids to owning tab for status-dot updates
- Helpers: `populateStudioRoster()`, `_fillStudioStudentPicker()`, `_renderStudioStudentInfo()`,
  `onStudentPicked()`, `clearAllStepDots()`, `stopPipeline()`, `setPipelineRunning()`

**Server (Python) key additions:**
- `PipelineReq.skills/languages/code_langs` — curriculum fields
- `PipelineReq.run_eval` / `run_postmortem` — honor the Studio checkboxes
- `PipelineReq.students_batch: list[dict]` — sequential batch; when non-empty,
  stages 4-11 loop per student. When empty, single-student back-compat path runs.
- `POST /api/pipeline/stop` — cancel endpoint; flips `_CANCEL_REQUESTED` and kills `_ACTIVE_PROC`
- `_popen()` registers each subprocess in a single-slot `_ACTIVE_PROC` under `_PIPELINE_LOCK`
- `_is_cancel_requested()` / `_reset_cancel_flag()` / `_request_cancel()` — cancel plumbing
- `_PROMPT_SKILL_MAP` — maps prompt ID prefixes to skill names
- `_filter_prompts_by_curriculum()` — filters prompts JSONL by skills + languages
- Pipeline uses `actual_prompts` (filtered) instead of `req.prompts_path` (original)

**SSE event types** (streamed from `/api/pipeline/start`):
- `stage {stage, status, msg}` — per-stage lifecycle (running/done/fail/skip/starting)
- `log {stage, text}` — line-by-line subprocess stdout
- `train_progress {stage, step, total, loss, epoch, eta}` — parsed from trainer_log
- `metrics {cpu, ram, gpu, vram, bottleneck, ...}` — bottom-rail sampler
- `student_begin {idx, total, name, hf_id, tag}` — batch: a new student starts stages 4-11
- `student_end {idx, total, name, tag, status}` — batch: student finished (done/fail/cancelled)
- `done_all {gold, silver, drop, report, batch_size, batch_status}` — pipeline complete
- `error {msg}` — fatal early-exit

**Crash guards:**
- `_responses_have_content()` — blocks if >95% teacher answers are empty
- GOLD=0 guard — blocks training when no consensus data
- Curriculum zero-match — blocks generation when filter keeps 0 prompts
- Missing SFT yaml — blocks training when config generation silently failed
- Always use `sys.executable` for subprocess calls, never bare `llamafactory-cli`

---

## Daily To-Do — Research & Enhancement Priorities

Key findings from literature review (arXiv 2405.09673, 2603.02224, 2401.05605,
2506.21035, 2311.06243, 2410.13025) that drive the roadmap below:

- LoRA forgets less per-task than full FT, but does **not** automatically prevent
  catastrophic forgetting in continual learning — replay buffers are mandatory.
- Rank matters: rank 8 is ~half the quality of rank 256 for code tasks. Minimum
  rank 64 for code/math. Forgetting is rank-invariant at high subspace angles.
- Skill composition (merging multiple LoRAs) is an unsolved frontier — every method
  trades off modularity vs. quality. No clean winner yet at 0.5B–3B scale.

### Priority 1 — Adaptive Rank Selection (rank probing) -- IMPLEMENTED

- [x] `scripts/rank_probe.py` — 50-step sweep at ranks [16,32,64,128], measures loss
  convergence slope, selects lowest rank within 5% of steepest (most negative) slope
- [x] Importable: `from rank_probe import probe_rank` returns `(best_rank, details_dict)`
- [x] Wired into `skill_branch.py --rank auto` — probes before training, graceful fallback
- [x] Writes `saves/_probes/{tag}_report.md` with per-rank comparison table
- [x] Also supports OFT probing via `_probe_oft_yaml()`
- [ ] **Next:** run actual probes on real trunk+skill data to validate slope heuristic
- **Status:** code complete, needs battle-testing on real data

### Priority 2 — OFT/BOFT Head-to-Head vs LoRA -- IMPLEMENTED

- [x] `skill_branch.py --adapter-type oft` — full OFT branch pipeline
- [x] `_branch_oft_yaml()` builder with `oft_rank`, `oft_block_size`, `oft_target`
- [x] LlamaFactory already supports `finetuning_type: "oft"` natively (confirmed in
  `src/llamafactory/hparams/finetuning_args.py` and `src/llamafactory/model/adapter.py`)
- [x] CLI flags: `--adapter-type lora|oft`, `--oft-rank` (default 8), `--oft-block-size` (default 32)
- [x] OFT merge note: `llamafactory-cli export` only supports LoRA; OFT adapters used
  via PEFT at inference or merged manually with `model.merge_and_unload()`
- [ ] **Next:** run LoRA vs OFT head-to-head on same skill data, compare quality + forgetting
- **Status:** code complete, needs head-to-head benchmark run

### Priority 3 — Gated Composition Benchmark -- IMPLEMENTED

- [x] `scripts/composition_bench.py` — 5-strategy benchmark framework:
  (0) Trunk baseline, (1) Individual branches, (2) DARE-TIES merge,
  (3) Adapter routing (keyword classifier + PEFT `set_adapter()`),
  (4) Weighted adapter stack (PEFT `add_weighted_adapter()`)
- [x] Perplexity on gold responses (prompt-masked cross-entropy) + tok/sec latency
- [x] Per-skill breakdown + overall comparison table in Markdown report
- [x] `--dry-run` validates inputs without loading models
- [x] Keyword-based skill classifier for routing (heuristic oracle; swap for trained
  classifier later)
- [ ] **Next:** train 2-3 branches, build mixed test JSONL, run the actual benchmark
- **Status:** code complete, needs trained branches + test data to run

### Priority 4 — Smart Replay Buffer Composition

- [ ] After trunk merge, compute sentence embeddings on consensus SFT data
- [ ] Run k-means clustering (k≈50) and store cluster assignments
- [ ] When building replay buffer for a branch, sample proportionally from clusters
- [ ] Test whether 5% smart-sampled replay matches 20% random replay
- **Goal:** reduce replay fraction needed → faster branch training, less data overhead

### Priority 5 — Curriculum-Aware Distillation (Phase 1 improvement)

- [ ] After teacher generation, score each example by student perplexity (pre-training)
- [ ] Sort easiest-first, train epoch 1 on easy 50%, epoch 2 on full set ("baby steps")
- [ ] Measure convergence speed and final quality vs. random-order baseline
- [ ] Especially impactful for 0.5B students that struggle with hard examples early
- **Goal:** faster convergence + better final quality in Phase 1 distillation
