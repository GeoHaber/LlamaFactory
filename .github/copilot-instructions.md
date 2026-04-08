# GitHub Copilot Instructions for LLaMA Factory

## What This Project Is

This is a **fork** of [hiyouga/LlamaFactory](https://github.com/hiyouga/LlamaFactory)
(upstream: `origin`; fork: `fork` remote → `GeoHaber/LlamaFactory`).

**LlamaFactory creates new GGUF models from scratch.** It does this by training one or
more small "Student" models under the supervision of multiple large "Expert" LLMs
(the teachers), coordinated by a "Dean" (Zena — a Gemma 4 GGUF chatbot that
cross-examines teachers, detects hallucinations, and picks the best graduates).

After training, the graduates are tested and verified — that verification reuses the
model comparison and benchmarking infrastructure from **LLM_TEST_BED / ZEN_Forge**,
the test environment where this LlamaFactory fork lives.

### The Three Projects

| Project | Repo / Location | Role |
|---------|-----------------|------|
| **LlamaFactory** (this repo) | `GeoHaber/LlamaFactory` fork of `hiyouga/LlamaFactory` | **The University.** Trains student models via SFT/DPO/LoRA under expert teacher supervision. Produces new GGUF models. |
| **LLM_TEST_BED / ZEN_Forge** | `GeoHaber/LLM_TEST_BED/ZEN_Forge/` (parent directory) | **The Exam Hall.** Side-by-side model comparison, LLM-as-judge scoring, stress testing, benchmarking. Already existed before LlamaFactory — used to verify graduates. |
| **zen_core_libs** | `GeoHaber/zen_core_libs` (separate repo, pip-installed) | **The Library.** Shared utilities: `InProcessAdapter` (GGUF inference), `SPSCRingBuffer` (lock-free FIFO), `RAMPressureThrottle`, evaluation functions (`compute_retention`, `graduation_report`, `bootstrap_ci`). |

### The University Metaphor

```
  Dean (Zena — Gemma 4 GGUF)
    |
    |-- credentials-checks each expert
    |-- cross-examines experts on each other's answers
    |-- judges hallucinations on borderline samples
    |
  Expert Teachers (3+ large GGUF LLMs, 7B-26B)
    |
    |-- all answer the same prompts concurrently
    |-- consensus answers (GOLD) become SFT training data
    |-- disagreements (SILVER) become DPO preference data
    |-- no majority (DROP) get discarded
    |
  Student Models (0.5B-3B, trained via LlamaFactory)
    |
    |-- learn from GOLD data (supervised fine-tuning)
    |-- learn from SILVER data (preference optimization)
    |-- multiple variants compete in the Forge Matrix
    |-- best variant = "Champion" → merged → exported to GGUF
    |
  Graduation (verified by ZEN_Forge test infrastructure)
    |
    |-- quick quiz (10 probes) → deep exam (45+ probes)
    |-- teacher baseline comparison (retention ratio)
    |-- GGUF export + speed benchmark
    |-- graduation dashboard (pass / fail / review)
```

### What We Modified in Upstream LlamaFactory

Our fork adds **30,044 lines** across **265 files** on top of upstream `main`. Changes:

| Area | Lines Added | What |
|------|-------------|------|
| `scripts/` (19 new scripts) | ~15,400 | Multi-teacher generation, purification, config gen, student forge, evaluation, dashboard, GGUF export, orchestrator, CLI |
| `src/llamafactory/webui/` (17 files) | ~2,400 | Hardware auto-tuner, AD coordinator, distillation UI tabs, GGUF export enhancements, inference panel, CSS, locales |
| `src/llamafactory/chat/` (2 new files) | ~440 | `autotune.py` (hardware auto-tuning), `coordinator.py` (adaptive decomposition) |
| `src/llamafactory/` (other core) | ~260 | API coordinator protocol, hparams JSON validation, minor fixes |
| `tests/` (11 test files) | ~1,500 | Forge auto-heal, integration pipeline, end-to-end toy, JSON resilience, coordinator, autotune |
| `data/`, `examples/` | ~5,000 | Zena007 prompts, teacher responses, purified datasets, forge matrix configs, distillation YAML examples |
| `.github/`, docs | ~400 | CI workflow, copilot-instructions, contributing guide |

Upstream LlamaFactory code was **not refactored or broken** — all changes are additive.
Training stages 5/6/7 call `llamafactory-cli train` and `llamafactory-cli export` directly.

## LlamaFactory Core (Upstream)

### Architecture Versions

Two parallel architectures, switched via `USE_V1` environment variable:

**v0 (default)**: `api`, `webui` -> `chat`, `eval`, `train` -> `data`, `model` -> `hparams` -> `extras`

**v1** (`USE_V1=1`): `trainers` -> `core` -> `accelerator`, `plugins`, `config` -> `utils`

### Code Structure (v0)

- `src/llamafactory/` - Main package
  - `api/` - OpenAI-style API
  - `chat/` - Chat interface + **autotune.py** (hardware auto-tuner) + **coordinator.py** (AD coordinator)
  - `cli.py` - Command-line interface
  - `data/` - Data processing and dataset handling
  - `eval/` - Model evaluation utilities
  - `extras/` - Additional utilities and helpers
  - `hparams/` - Hyperparameter definitions
  - `model/` - Model loading, patching, quantization
  - `train/` - Training pipeline (SFT, DPO, PPO, RM, PT, KTO, ORPO)
  - `webui/` - Gradio-based web interface (LLaMA Board) + distillation UI
- `src/train.py` - Training entry -> `llamafactory.train.tuner`
- `src/webui.py` - Web UI entry -> `llamafactory.webui.interface`
- `src/api.py` - API server entry -> `llamafactory.api.app`

### Code Structure (v1, `USE_V1=1`)

- `src/llamafactory/v1/` - `trainers/` -> `core/` -> `accelerator/`, `plugins/`, `config/` -> `utils/`

## Development Practices

### Code Style

- Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- Use ruff for linting and formatting
- Line length: 119 characters
- Indentation: 4 spaces
- Quote style: double quotes
- Use Google-style docstrings for documentation

### Import Organization

- Known first-party: `llamafactory`
- Known third-party: `accelerate`, `datasets`, `gradio`, `numpy`, `peft`, `torch`, `transformers`, `trl`
- Use 2 blank lines after imports

### Quality Checks

Before committing code, run:
```bash
make style      # Auto-fix style issues
make quality    # Check code quality
make test       # Run test suite
```

Or use the combined command:
```bash
make commit     # Run pre-commit hooks
```

#### X_Ray_LLM Code Quality Scan

All scripts pass [X_Ray_LLM](https://github.com/GeoHaber/X_Ray_LLM) (78 rules, 53 files)
with **0 HIGH, 0 MEDIUM, 0 LOW** findings. Run the scan with:

```bash
cd ../X_Ray_LLM
python -m xray.agent "../LLM_Factory/scripts" --dry-run --exclude "distill_ui\.py" "distill_ui_old\.py"
```

Security hardening applied:
- **XSS prevention**: `esc()` HTML sanitizer in `distill.html` — all dynamic innerHTML uses it
- **JSON resilience**: All `json.loads()`/`json.load()` calls wrapped in `try/except`
- **Narrow exceptions**: No bare `except Exception` — all catches specify exact error types
- **No weak hashes**: SimHash uses SHA-256 (not MD5)
- **No mutable globals**: Singleton patterns use `_DeanHolder`/`_EmbedHolder` classes
- **Suppressed false-positives**: `# xray: ignore[SEC-015]` on verified pip-installed packages

### Testing

- Use pytest for testing
- Tests are located in `tests/` and `tests_v1/` directories
- Run tests with: `make test` (which runs `WANDB_DISABLED=true pytest -vv --import-mode=importlib tests/ tests_v1/`)
- Disable wandb during testing to avoid external dependencies
- **Note**: Training configurations require GPU machines, so training is typically not tested end-to-end. Use `make test` to validate file-level functionality.

### Building

Build the package with:
```bash
pip3 install build && python3 -m build
```

### License

- All source files must include the Apache 2.0 license header
- Check license headers with: `make license`

## Common Patterns

### Configuration Files

- Training configurations are typically YAML or JSON files in `examples/` directory
- Hyperparameters are defined using dataclasses in `src/llamafactory/hparams/`

### Model Support

- New model support is added through model patches in `src/llamafactory/model/`
- Visual models use the visual utilities in `src/llamafactory/model/model_utils/visual.py`
- Quantization support is in `src/llamafactory/model/model_utils/quantization.py`

### Data Processing

- Dataset definitions are in `data/dataset_info.json`
- Data templates and processors are in `src/llamafactory/data/`

### Training

- Training pipelines are in `src/llamafactory/train/`
- Support for different training methods: SFT, DPO, PPO, RM, PT, KTO, ORPO

### Multi-Teacher Distillation Pipeline (The University)

LlamaFactory creates new GGUF models by running students through a supervised
distillation pipeline. The Experts (large GGUF teachers) generate training data,
the Dean (Zena) quality-controls it, and LlamaFactory's own training CLI produces
the graduates. After graduation, the test infrastructure from ZEN_Forge / LLM_TEST_BED
verifies the students against the teachers.

#### Roles

| Role | What | Example |
|------|------|---------|
| **Dean (Zena)** | Gemma 4 GGUF chatbot. Credentials-checks experts, cross-examines answers, judges hallucinations on borderline SILVER samples. | `scripts/zena_dean.py` |
| **Experts (Teachers)** | 3+ large GGUF models (7B-26B) that all answer the same prompts. Their consensus becomes training data. | Gemma-4-26B, Qwen2.5-14B, Mistral-7B |
| **Students** | Small models (0.5B-3B) trained via LlamaFactory LoRA SFT/DPO on expert consensus. Multiple variants compete. The champion gets merged and exported to GGUF. | Qwen2.5-1.5B-Instruct |
| **Exam Hall (ZEN_Forge)** | LLM_TEST_BED's existing comparison/benchmarking infrastructure. Reused for post-training verification: side-by-side eval, LLM-as-judge, stress testing. | `comparator_backend.py`, `zen_eval.py`, `benchmark_14_models.py` |

#### Key scripts

| Script | Role | Purpose |
|--------|------|---------|
| `scripts/zena_dean.py` | Dean | Teacher credential checks, cross-examination, peer-respect matrix, hallucination judging |
| `scripts/multi_teacher_generate.py` | Experts | Concurrent GGUF teacher response generation via SPSC ring-buffer FIFO |
| `scripts/hallucination_gates.py` | Dean + Experts | 5-gate hallucination detection chain (self-consistency, semantic drift, fact grounding, confidence, Zena cross-exam) |
| `scripts/purify_teacher_outputs.py` | Dean | Split expert responses into GOLD (SFT) / SILVER (DPO) / DROP |
| `scripts/gen_distill_configs.py` | — | Auto-generate LlamaFactory-compatible SFT / DPO / merge YAML configs |
| `scripts/run_student_forge.py` | Students | Parallel training of multiple student variants (Forge Matrix) |
| `scripts/eval_student_panel.py` | Exam | Two-pass evaluation: quick quiz -> deep exam + GGUF teacher baseline |
| `scripts/graduation_dashboard.py` | Exam | HTML dashboard with SVG verdict ring (pass / fail / review) |
| `scripts/slim_down.py` | Export | GGUF conversion + speed benchmark + Pareto frontier |
| `scripts/student_registry.py` | Registry | Persist eval scores and gap analysis across runs |
| `scripts/validate_datasets.py` | QA | Data validation: duplicates, leakage, balance, DPO validity |
| `scripts/prompt_difficulty.py` | QA | Prompt difficulty scoring + histogram + filtering |
| `scripts/bayesian_forge.py` | Tuning | Bayesian hyperparameter search (Optuna TPE sampler) |
| `scripts/teacher_profile.py` | QA | Per-teacher quality analysis (agreement, GOLD/SILVER contribution) |
| `scripts/pipeline_preflight.py` | QA | Pre-run validator (manifest, prompts, configs, disk, deps) |
| `scripts/loss_chart.py` | Viz | Loss comparison chart (text + SVG) with convergence prediction |
| `scripts/pipeline_events.py` | Logging | Structured JSON event logger for pipeline stages |
| `scripts/orchestrate_pipeline.py` | Orchestrator | Cross-platform Python 11-stage pipeline (replaces PS1) |
| `scripts/zenforge_cli.py` | CLI | Unified entry point (`python scripts/zenforge_cli.py <command>`) |
| `scripts/distill_ui.py` | UI | Gradio-based distillation wizard (phase 1/2/3) — **deprecated, replaced by distill_server.py** |
| `scripts/distill_server.py` | UI | FastAPI server (port 7870) + SSE pipeline streaming + curriculum filtering |
| `scripts/distill.html` | UI | Single-page distillation wizard (served by distill_server.py) |
| `scripts/benchmark_multi_teacher_dispatch.py` | Bench | A/B benchmark for dispatch modes |
| `scripts/run_zena007_end_to_end.ps1` | Orchestrator | PowerShell full pipeline (sequential or Forge Matrix) |

#### Distillation Web UI (`distill_server.py` + `distill.html`)

The web UI is a standalone FastAPI + vanilla HTML/JS app that replaces the earlier Gradio
`distill_ui.py`. It provides a 3-phase wizard:

1. **Staff & Faculty** — Pick Dean (Zena GGUF) + 3+ expert teachers from local model scan
2. **Brain Architect** — Skill-first student design: pick skills (translation, coding, OCR,
   chat, reasoning, math, summarize, creative), configure languages, system recommends
   quality tier (OK / Good / Excellent mapping to hidden 0.5B/1.5B/3B HF models)
3. **Teaching & Graduation** — SSE-streamed 10-stage pipeline with live progress

**Curriculum filtering** is the key innovation: `_filter_prompts_by_curriculum()` filters the
prompts JSONL file to keep only prompts matching the student's enrolled skills and languages
before generation begins. Prompt IDs encode their category (`tr-en2es-0013` = translation
English-to-Spanish, `chat-en-0524` = conversation, `ocr-0001` = OCR cleanup). The filtered
file is saved as `curriculum_prompts.jsonl` and used for all downstream stages. A
`curriculum.json` is persisted to `saves/<tag>/` for the graduation dashboard.

**Server endpoints:**
- `GET /` — serves `distill.html`
- `GET /api/scan` — GGUF model discovery with arch/role/quant metadata
- `POST /api/pipeline/start` — SSE stream for the full pipeline
- `POST /api/chat` — Dean chat relay
- `POST /api/credential-check` / `POST /api/cross-exam` — Dean examinations
- `GET /api/progress` — generation progress polling
- `POST /api/gen-configs` — YAML config generation

**Run:** `python scripts/distill_server.py` (requires Python >= 3.14)

#### The 11-Stage Pipeline

```
  Stage 1: GENERATE        Experts answer prompts concurrently
      |                    (SPSC ring-buffer FIFO, RAM-pressure throttle)
      v
  Stage 2: PURIFY          Dean judges: GOLD -> SFT, SILVER -> DPO, DROP
      |                    (hallucination gates, synthetic DPO if needed)
      v
  Stage 3: VALIDATE        Check data quality (duplicates, leakage, balance)
      v
  Stage 4: CONFIGURE       Auto-generate LlamaFactory YAML configs
      v
  Stage 5: SFT TRAIN       LlamaFactory trains student on GOLD consensus
      |                    (llamafactory-cli train, LoRA, auto-resume)
      v
  Stage 6: DPO TRAIN       LlamaFactory trains student on SILVER conflicts
      |                    (optional — skipped when no DPO data)
      v
  Stage 7: MERGE           LlamaFactory merges LoRA adapters into base model
      |                    (llamafactory-cli export)
      v
  Stage 8: FORGE BRIDGE    Synthesize results + champion metadata
      v
  Stage 9: EVALUATE        Exam Hall verifies student vs all teachers
      |                    (quick quiz -> deep exam, GGUF teacher eval)
      v
  Stage 10: GGUF EXPORT    Convert merged model to GGUF + speed benchmark
      v
  Stage 11: QUALITATIVE    Generate sample responses for human review
      v
  GRADUATION DASHBOARD     Pass / Fail / Review verdict
```

Every stage is **idempotent** — re-run after any crash and completed stages auto-skip.

#### Generation architecture (v4 — SPSC Ring-Buffer FIFO)

Expert teachers run concurrently via lock-free ring buffers:

- Per-teacher `SPSCRingBuffer` (lock-free, GIL-atomic integer stores)
- Default 2048-slot depth via `--fifo-size 0` (auto mode)
- `RAMPressureThrottle` with hysteresis (`--ram-pause-pct 12 --ram-resume-pct 22`)
- Adaptive decoding budgets per prompt category (`--adaptive-budgets`)
- Per-teacher JSONL checkpoints in `data/<tag>/checkpoints/` for crash-safe resume
- Backends: `InProcessAdapter` (direct GGUF, no HTTP) or `LlamaServerManager` (HTTP)

Shared library: `zen_core_libs` provides `SPSCRingBuffer`, `InProcessAdapter`,
and `RAMPressureThrottle` in `zen_core_libs.common.system` and `zen_core_libs.llm`.

`InProcessAdapter.chat()` signature: `messages: list[dict[str, Any]] | None = None` — always
use fully-typed dicts. The `_build_messages` helper also returns `list[dict[str, Any]]`.

#### Pipeline stage skip conditions (crash-resume)

Every stage is idempotent — re-run the same command after any crash:

| Stage | Skip condition | Resume behaviour |
|-------|---------------|-----------------|
| 1. Generate | `teacher_responses.jsonl` row count == prompt count | Per-teacher checkpoints auto-merged; missing prompts refilled |
| 2. Purify | `purified/purification_report.json` exists | Full skip; `--synthetic-dpo` generates cross-prompt DPO pairs |
| 3. Validate | N/A | Always runs (fast) |
| 4. Configure | Both `*_sft.yaml` and `*_merge.yaml` exist | Full skip |
| 5. SFT Train | `_is_training_complete()` returns True (checks trainer_log.jsonl) | Resumes from highest `checkpoint-N`; `overwrite_output_dir: false` |
| 6. DPO Train | Same as SFT, or no `conflict_dpo.jsonl` samples | Skipped when no DPO data |
| 7. Merge | `saves/<tag>/merged/config.json` exists | Full skip |
| 8. Forge Bridge | `saves/<tag>/forge_results.jsonl` exists | Full skip |
| 9. Evaluate | `saves/<tag>/eval_scorecards.jsonl` exists | Full skip; GGUF teacher eval via InProcessAdapter |
| 10. GGUF Export | `saves/<tag>/gguf/slim_down_results.jsonl` exists | Full skip; exports Q4_K_M + speed bench |
| 11. Qualitative | `saves/<tag>/qualitative_eval.jsonl` exists | Full skip |
| Dashboard | `saves/<tag>/graduation_report.json` exists | Exports markdown + HTML |

#### Student Forge Matrix (parallel multi-variant training)

`run_zena007_end_to_end.ps1 -UseForge` activates Forge Matrix mode which trains N student
variants in parallel (controlled by `max_parallel` in the matrix YAML). Each variant can
differ in model, LoRA rank, learning rate, and epoch count. After training, an eval panel
scores all variants and selects the champion for merging.

Matrix config: `data/forge_matrix/<tag>_matrix.yaml` — defines `sft_data`, `dpo_data`,
`max_parallel`, `eval_probe_split`, and per-variant settings.

The forge automatically:
1. Splits `consensus_sft.jsonl` into `train_sft.jsonl` + `eval_probes.jsonl` (probe_fraction)
2. Registers `<tag>_forge_train_sft` in `dataset_info.json` (relative to `data/` dir)
3. Generates per-variant YAML configs in `examples/distillation/auto/forge/<tag>/`
4. Trains variants concurrently via SPSC ring-buffer result collection
5. Writes `saves/<tag>/forge_results.jsonl` and `saves/<tag>/champion.txt`

#### Student Forge Auto-Healing (`run_student_forge.py`)

`ForgeState` provides crash-safe state management for multi-day training runs:

- **Atomic state file** (`saves/<tag>/forge_state.json`) — records completed/failed variants,
  heartbeat timestamps, and partial results. Survives crashes.
- **Checkpoint resume** — `_find_latest_checkpoint(output_dir)` finds the highest
  `checkpoint-N` directory for automatic resume via `overwrite_output_dir: False`.
- **Heartbeat** — background thread writes timestamps every 30s; stale heartbeats
  (>120s) indicate a hung worker.
- **LLM diagnosis** — `_diagnose_with_llm(stderr)` sends error logs to a local GGUF
  model for root-cause analysis when a training variant fails.
- **Idempotent re-run** — ForgeState skips already-completed variants on restart.

Tests: `tests/test_forge_autoheal.py` (12 tests covering state, checkpoints, heartbeat,
atomic writes).

#### Graduation Dashboard (`graduation_dashboard.py`)

Generates a single-page HTML report from a graduation report JSON:

- SVG ring with pass/fail/review verdict and percentage
- Clean HTML table with per-category retention, confidence, and status
- Ruin and emergence alert banners
- Raw JSON in an accordion for debugging

Usage: `python scripts/graduation_dashboard.py --saves-tag zena007`

#### Graduation Exam (`zen_core_libs.llm.eval`)

Shared evaluation library in `zen_core_libs` (not in this repo). Key exports:

- `compute_retention()` — teacher-vs-student retention ratio per category
- `graduation_report()` — multi-student comparison with pass/fail verdicts
- `bootstrap_ci()` — confidence intervals via bootstrap resampling
- `detect_emergence()` — finds categories where student beats all teachers
- `RuinDetected` — exception raised when retention drops below threshold
- `extract_category()` — extracts category from prompt ID or explicit field

Tests: `zen_core_libs/llm/tests/test_eval.py` (42 tests).

#### Config generation rules (`gen_distill_configs.py`)

- `_merge_config(student, tag, has_dpo=True/False)` — only chains DPO adapter when
  `has_dpo=True`; if no conflict/DPO samples exist the merge config uses SFT adapter only
- **Do NOT add `booster: auto`** (or any non-HfArgumentParser key) to generated YAML configs —
  LlamaFactory's `parse_dict` will raise `ValueError: Some keys are not used` at startup
- `bf16` is set to `not cpu_safe` — always pass `--cpu-safe` on CPU-only machines

#### `dataset_info.json` file_name convention

Paths in `dataset_info.json` are **relative to `dataset_dir`** (default `"data"`). Always
strip the leading `data/` prefix. Use `_rel_to_data(path)` helper in `run_student_forge.py`
to compute the correct relative path. Wrong: `"data/zena007/purified/x.jsonl"` →
LlamaFactory joins `data/ + data/zena007/...` = double-path crash.

#### Advanced purification features (`purify_teacher_outputs.py`)

- **Embedding-based reasoning**: `--use-embeddings` uses all-MiniLM-L6-v2 for semantic
  similarity (requires `sentence-transformers`). Falls back to SimHash when unavailable.
- **Teacher weighting**: `--teacher-weights '{"qwen": 1.5, "llama": 1.0}'` — weighted
  majority voting. Higher weight = more influence in consensus detection.
- **Auto-tune thresholds**: `--auto-tune` sweeps answer/reason thresholds via grid search
  to hit a target GOLD percentage (default 60%). Controlled by `--auto-tune-target`.
- **Synthetic DPO**: `--synthetic-dpo` generates cross-prompt DPO pairs from GOLD samples
  by pairing highest-confidence chosen with lowest-confidence rejected.
- **Curriculum learning**: `--curriculum` sorts GOLD output by difficulty (easy -> hard)
  for progressive training.

#### Early stopping (`gen_distill_configs.py`)

`--early-stopping-patience N` (N > 0) adds `eval_strategy`, `eval_steps`, and
`load_best_model_at_end` to SFT/DPO configs. Monitors `eval_loss` and stops when
it doesn't improve for N eval rounds.

#### Native GGUF export (`slim_down.py`)

`--llama-cpp-path /path/to/llama.cpp` enables direct GGUF conversion via llama.cpp's
`convert_hf_to_gguf.py` + `llama-quantize`. Falls back to LlamaFactory export CLI
when not specified.

#### CI/CD

`.github/workflows/ci.yml` runs on push/PR: 59 tests (3 suites), ruff lint, and script
smoke tests. No GPU required.

## Key Dependencies

- Python >= 3.9.0
- PyTorch and transformers for model handling
- datasets for data processing
- peft for parameter-efficient fine-tuning
- accelerate for distributed training
- gradio for web UI
- trl for reinforcement learning
- psutil for RAM-pressure throttling (multi-teacher generation)
- Optional: vllm/sglang for inference, flash-attention-2, unsloth, liger-kernel

## Entry Points

- **CLI Training**: `llamafactory-cli train --config examples/train_lora/llama3_lora_sft.yaml`
- **Web UI**: `llamafactory-cli webui` or `python src/webui.py`
- **API Server**: `llamafactory-cli api` or `python src/api.py`
- **Chat Interface**: `llamafactory-cli chat --model_name_or_path MODEL_PATH`
- **Distillation Web UI**: `python scripts/distill_server.py` (port 7870, Python >= 3.14)
- **Multi-Teacher Generation**: `python scripts/multi_teacher_generate.py --manifest MANIFEST --prompts PROMPTS --dispatch-mode teacher-fifo --fifo-size 0`
- **Dispatch Benchmark**: `python scripts/benchmark_multi_teacher_dispatch.py --manifest MANIFEST --prompts PROMPTS --output-dir OUT`
- **End-to-End Distillation (sequential)**: `./scripts/run_zena007_end_to_end.ps1`
- **End-to-End Distillation (Forge Matrix)**: `./scripts/run_zena007_end_to_end.ps1 -UseForge`
- **Forge dry-run**: `python scripts/run_student_forge.py --matrix data/forge_matrix/zena007_matrix.yaml --tag zena007 --dry-run`
- **Eval panel only**: `python scripts/eval_student_panel.py --saves-tag zena007 --probes data/zena007/purified/eval_probes.jsonl`
- **Graduation dashboard**: `python scripts/graduation_dashboard.py --saves-tag zena007`

## Environment Setup

For development:
```bash
pip install -e ".[dev]"
```

## Important Notes

- The project supports multiple backends: default PyTorch, vLLM, SGLang
- Megatron-core training is supported via mcore_adapter
- SwanLab and W&B are supported for experiment tracking
- Docker support is available with pre-built images
- Day-0/Day-1 support for latest cutting-edge models
- Multi-modal support for vision and audio understanding tasks

## Contribution Guidelines

1. Fork the repository
2. Create a development branch
3. Set up development environment with `pip install -e ".[dev]"`
4. Make changes following the style guide
5. Run quality checks: `make style && make quality`
6. Run tests: `make test`
7. Submit a pull request

### WebUI Architecture

The Web UI is built with Gradio in `src/llamafactory/webui/`. Key files:

- `interface.py` -- top-level `create_ui()` assembles tabs, JS injection, Zena menu
- `engine.py` -- `Engine` class: state manager, coordinates manager/runner/chatter
- `runner.py` -- `Runner` class: training/eval subprocess lifecycle
- `chatter.py` -- `WebChatModel`: model load/unload, chat streaming
- `control.py` -- dropdown/validation handlers (model info, checkpoints, auto-tune)
- `components/` -- one file per tab (top, train, eval, infer, export, chatbot, data, footer)

All visible buttons are wired to real handlers. 6 help accordions (`*_help_tab`) are
`visible=False` -- they have `lang.change` handlers but are never shown (JS `?` icons
serve the same purpose). No NOOP or dead handlers exist.

### Testing

Local test suites (no GPU required):

| Suite | Command | Tests |
|-------|---------|-------|
| Forge auto-heal | `pytest tests/test_forge_autoheal.py -v --noconftest` | 16 |
| Integration pipeline | `pytest tests/test_integration_pipeline.py -v --noconftest` | 35 |
| End-to-end toy | `pytest tests/test_end_to_end_toy.py -v --noconftest` | 8 |
| Graduation eval (zen_core_libs) | `pytest zen_core_libs/llm/tests/test_eval.py -v` | 42 |
| **Total** | | **101** |

## Common Commands

- `make style` - Format code
- `make quality` - Run linters
- `make test` - Run tests
- `make commit` - Install and run pre-commit hooks
- `make license` - Check license headers
