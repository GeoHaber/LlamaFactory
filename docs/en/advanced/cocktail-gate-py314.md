# Cocktail Gate Environment (Python 3.14)

This runbook is for the full gate pipeline:

1. Shadow smoke SFT + DPO gate (fast fail)
2. Full SFT distillation
3. Full DPO distillation
4. Baseline benchmark compare
5. Coordinator benchmark compare

## Python 3.14 baseline

This workflow is now Python 3.14-focused with a validated package set:

- `torch==2.11.0`
- `torchvision==0.26.0`
- `torchaudio==2.11.0`
- `transformers==4.57.6`
- `datasets==4.8.4`
- `dill==0.4.1`
- `multiprocess==0.70.19`

These versions are now aligned with LLaMA Factory dependency checks, so `DISABLE_VERSION_CHECK=1` is not required.

## Setup

From repo root:

```powershell
pwsh -File scripts/setup_cocktail_gate_py314.ps1
```

Optional CPU-only setup:

```powershell
pwsh -File scripts/setup_cocktail_gate_py314.ps1 -CpuOnly
```

## Run full gate

```powershell
pwsh -File scripts/run_cocktail_gate.ps1 `
  -PythonExe ".venv-py314/Scripts/python.exe" `
  -RequirePython314 `
  -ApiBase "http://127.0.0.1:8000/v1" `
  -ModelName "Qwen/Qwen2.5-Coder-14B-Instruct"
```

The runner now includes:

- Automatic resume from latest checkpoint for full SFT/DPO (`checkpoint-*` under output_dir)
- Config preflight validator before each train stage
- Adaptive retry policy by failure type
- Shadow smoke gate before full distillation
- Automatic endpoint compatibility probe (`chat/completions` vs `completions`) for benchmark steps

## Reliability and self-healing options

Enable retry + fixer loop with your small chatbot:

```powershell
pwsh -File scripts/run_cocktail_gate.ps1 `
  -PythonExe ".venv-py314/Scripts/python.exe" `
  -RequirePython314 `
  -EnableSelfHeal `
  -SelfHealApiBase "http://127.0.0.1:8001/v1" `
  -SelfHealModel "your-small-chat-model" `
  -ApiBase "http://127.0.0.1:8000/v1" `
  -ModelName "Qwen/Qwen2.5-Coder-14B-Instruct"
```

Notes:

- `-EnableSelfHeal` runs a fixer loop (`scripts/gate_fixer_loop.py`) that emits structured JSON action plans and applies only guarded safe fixes.
- If `SelfHealApiBase` and `SelfHealModel` are set, the fixer loop merges a validated safe subset of LLM-proposed actions.
- The advisor (`scripts/gate_self_heal_advisor.py`) still runs to provide readable remediation hints.
- Step logs are written to `benchmark_output/cocktail_gate_logs/`.

Guardrails for auto-fix actions:

- Allowed action types: `set_yaml_key`, `ensure_dataset_columns`
- Allowed YAML keys: `per_device_train_batch_size`, `gradient_accumulation_steps`, `lora_target`, `preprocessing_num_workers`, `dataloader_num_workers`
- Allowed dataset column keys: `prompt`, `query`, `response`, `chosen`, `rejected`, `messages`

Adaptive retry profiles:

- Endpoint/network failures: longer backoff and more attempts
- Data/schema failures: fail fast with targeted hints
- OOM failures: apply one safe batch-size/grad-acc reduction and retry once

Endpoint mode behavior:

- Default: `-EndpointMode auto` (runner probes endpoint and selects `chat` or `completions`)
- Override manually if needed: `-EndpointMode chat` or `-EndpointMode completions`

## Preflight only

Run lightweight validation without starting training:

```powershell
C:/Users/dvdze/AppData/Local/Python/pythoncore-3.14-64/python.exe scripts/gate_preflight.py --config examples/distillation/student_sft_distill.yaml --dataset-info data/dataset_info.json
```

Disable preflight in emergency/debug flows:

```powershell
pwsh -File scripts/run_cocktail_gate.ps1 -NoPreflight ...
```

## Endpoint-first benchmark rerun

If distillation artifacts already exist and you only want benchmark comparisons:

```powershell
pwsh -File scripts/run_cocktail_gate.ps1 `
  -PythonExe ".venv-py314/Scripts/python.exe" `
  -RequirePython314 `
  -SkipTrain `
  -ApiBase "http://127.0.0.1:8000/v1" `
  -ModelName "Qwen/Qwen2.5-Coder-14B-Instruct"
```

## Artifacts

- Logs: `benchmark_output/cocktail_gate_logs/`
- Baseline benchmark outputs: `benchmark_output/cocktail_gate_baseline/...`
- Coordinator benchmark outputs: `benchmark_output/cocktail_gate_coordinator/...`
- Distillation outputs:
  - `saves/qwen2.5-coder-14b/lora/distill_sft`
  - `saves/qwen2.5-coder-14b/lora/distill_dpo`
