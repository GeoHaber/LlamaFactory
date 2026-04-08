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

Standalone FastAPI app (not Gradio) on port 7870. Requires Python >= 3.14.

**Run:** `python scripts/distill_server.py`

Three-phase wizard:
1. **Staff & Faculty** — pick Dean (Zena GGUF) + 3+ expert teacher GGUFs
2. **Brain Architect** — skill-first student design: pick skills (translation, coding, OCR,
   chat, reasoning, math, summarize, creative), configure languages, system recommends
   quality tier (OK/Good/Excellent → hidden 0.5B/1.5B/3B HF models)
3. **Teaching & Graduation** — SSE-streamed pipeline with 10 stage indicators + training progress

**Curriculum filtering** (`_filter_prompts_by_curriculum()`): before generation, prompts JSONL
is filtered to keep only prompts matching the student's enrolled skills and languages. Prompt
IDs encode category (`tr-en2es-0013` = translation, `chat-en-0524` = conversation, `ocr-0001` = OCR).
Filtered file → `curriculum_prompts.jsonl`; metadata → `saves/<tag>/curriculum.json`.

**Key data structures (JS):**
- `SKILLS[]` — 8 skills with id, icon, name, complexity
- `BRAIN_TIERS[]` — 3 quality tiers (OK/Good/Excellent) → HF model IDs
- `_selectedSkills`, `_selectedLangs`, `_selectedCodeLangs` — Sets of enrolled choices
- `_availableTiers[]` — computed by `recalcTiers()` based on skill complexity
- Student object: `{ name, hf_id, quality, skills, languages, code_langs, ... }`

**Server (Python) key additions:**
- `PipelineReq.skills/languages/code_langs` — curriculum fields
- `_PROMPT_SKILL_MAP` — maps prompt ID prefixes to skill names
- `_filter_prompts_by_curriculum()` — filters prompts JSONL by skills + languages
- Pipeline uses `actual_prompts` (filtered) instead of `req.prompts_path` (original)

**Crash guards:**
- `_responses_have_content()` — blocks if >95% teacher answers are empty
- GOLD=0 guard — blocks training when no consensus data
- Curriculum zero-match — blocks generation when filter keeps 0 prompts
- Missing SFT yaml — blocks training when config generation silently failed
- Always use `sys.executable` for subprocess calls, never bare `llamafactory-cli`
