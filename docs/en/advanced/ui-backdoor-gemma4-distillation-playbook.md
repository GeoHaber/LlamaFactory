# UI-Backdoor Distillation Playbook (Gemma -> Small 4-Language OCR+Chat)

This guide runs a full distillation exercise without pressing WebUI buttons.
It still uses backend logic that the WebUI uses under the hood.

## Goal

Build a smaller student model specialized for:

1. Translation for English, Romanian, French, Hungarian.
2. OCR understanding and cleanup.
3. General chat in those same four languages.
4. Refusal or redirection for other languages.

## What "UI-backdoor" means here

The Train tab in WebUI eventually reaches backend utilities in [src/llamafactory/webui/runner.py](src/llamafactory/webui/runner.py) and [src/llamafactory/webui/common.py](src/llamafactory/webui/common.py):

1. Build args.
2. Preview command text.
3. Save a training args yaml.
4. Launch `llamafactory-cli train` with that yaml.

This playbook uses [scripts/ui_backdoor_train.py](scripts/ui_backdoor_train.py), which intentionally uses the same backend helpers:

1. `gen_cmd` for preview text.
2. `save_cmd` for saving train args yaml.
3. `llamafactory-cli train` for launch.

No UI clicks are required.

## Efficient mode (recommended for future runs)

Use one fixed local interpreter for all stages and avoid repeated environment tooling calls.

Local validation pipeline (already working in this repo):

```powershell
./scripts/run_distill_local_no_uv.ps1 -RunAll
```

Target pipeline with your production templates:

```powershell
./scripts/run_distill_target_no_uv.ps1 -RunAll
```

Stage-by-stage use is also supported:

```powershell
./scripts/run_distill_local_no_uv.ps1 -RunSft
./scripts/run_distill_local_no_uv.ps1 -RunDpo
./scripts/run_distill_local_no_uv.ps1 -RunMerge
./scripts/run_distill_local_no_uv.ps1 -RunEval
```

Why this is efficient:

1. Single interpreter path: `.venv-py314/Scripts/python.exe`.
2. No repeated Python/venv setup chatter.
3. Same commands every run for reproducibility.

## Files prepared for this exercise

1. SFT template: [examples/distillation/gemma4_student_sft_ui_backdoor_template.yaml](examples/distillation/gemma4_student_sft_ui_backdoor_template.yaml)
2. DPO template: [examples/distillation/gemma4_student_dpo_ui_backdoor_template.yaml](examples/distillation/gemma4_student_dpo_ui_backdoor_template.yaml)
3. Merge template: [examples/merge_lora/gemma4_student_merge_ui_backdoor_template.yaml](examples/merge_lora/gemma4_student_merge_ui_backdoor_template.yaml)

## Step 1: Pick teacher and student

### Step 0.5: (Optional) Multi-Teacher Consensus Generation

For higher-quality training data, generate responses from multiple teachers in parallel
using FIFO dispatch, then split into consensus (SFT) and conflict (DPO) datasets:

```powershell
.venv-py314/Scripts/python.exe scripts/multi_teacher_generate.py `
  --manifest data/zena007/teacher_manifest.json `
  --prompts data/zena007_prompts.jsonl `
  --out data/zena007/teacher_responses.jsonl `
  --adaptive-budgets `
  --dispatch-mode teacher-fifo `
  --fifo-size 0 `
  --ram-pause-pct 12 --ram-resume-pct 22
```

Architecture notes:
- `--fifo-size 0` enables auto mode which provisions 2048-slot SPSC ring buffers per teacher (lock-free, zero synchronization overhead).
- `--ram-pause-pct 12 --ram-resume-pct 22` enables hysteresis-based RAM-pressure throttling for safe concurrent GGUF loading.
- `--adaptive-budgets` selects `max_tokens` and `temperature` per prompt category.
- Per-teacher JSONL checkpoints in `data/zena007/checkpoints/` enable crash-safe resume.

After generation, run the purifier:

```powershell
.venv-py314/Scripts/python.exe scripts/purify_teacher_outputs.py `
  --input data/zena007/teacher_responses.jsonl `
  --out-dir data/zena007/purified `
  --answer-threshold 0.85 --reason-threshold 0.6
```

This produces `consensus_sft.jsonl` and `conflict_dpo.jsonl` for the SFT and DPO stages below.

See [examples/distillation/README.md](../../../examples/distillation/README.md) for full FIFO dispatch architecture details.

## Step 1: Pick teacher and student

Recommended practical pairing:

1. Teacher: a strong Gemma-class model you trust for OCR+translation+chat quality.
2. Student: 1.5B to 3B instruct model.

Why this range:

1. 1.5B is fast and compact, but lower quality ceiling.
2. 3B is a better quality compromise for OCR plus multilingual chat.

## Step 2: Build distillation datasets

Create two datasets in [data/dataset_info.json](data/dataset_info.json) and corresponding files in [data](data):

1. `distill_ocr_chat_translate_4lang_train`
2. `distill_ocr_chat_translate_4lang_pref`

SFT dataset requirements:

1. Mix tasks roughly 40% translation, 35% OCR, 25% chat.
2. Keep only English, Romanian, French, Hungarian user prompts.
3. Add explicit examples where unsupported languages are declined.

DPO dataset requirements:

1. Pairs should reward strict language policy and OCR faithfulness.
2. Include hard negatives where model drifts into unsupported language.
3. Include OCR negatives with hallucinated text and formatting corruption.

Concrete files already wired in this repo:

1. SFT data: [data/distill_ocr_chat_translate_4lang_train.jsonl](data/distill_ocr_chat_translate_4lang_train.jsonl)
2. DPO data: [data/distill_ocr_chat_translate_4lang_pref.jsonl](data/distill_ocr_chat_translate_4lang_pref.jsonl)
3. Registry: [data/dataset_info.json](data/dataset_info.json)

## Step 3: Edit templates for your run

Update these keys before launching:

1. In [examples/distillation/gemma4_student_sft_ui_backdoor_template.yaml](examples/distillation/gemma4_student_sft_ui_backdoor_template.yaml):
   1. `model_name_or_path`
   2. `dataset`
   3. `template`
   4. `max_samples`
2. In [examples/distillation/gemma4_student_dpo_ui_backdoor_template.yaml](examples/distillation/gemma4_student_dpo_ui_backdoor_template.yaml):
   1. `model_name_or_path`
   2. `adapter_name_or_path`
   3. `dataset`
   4. `template`
3. In [examples/merge_lora/gemma4_student_merge_ui_backdoor_template.yaml](examples/merge_lora/gemma4_student_merge_ui_backdoor_template.yaml):
   1. `model_name_or_path`
   2. `adapter_name_or_path`
   3. `template`

## Step 4: Preview commands through the UI backend path

PowerShell:

```powershell
.venv-py314/Scripts/python.exe scripts/ui_backdoor_train.py `
  --config examples/distillation/gemma4_student_sft_ui_backdoor_template.yaml `
  --preview-only
```

This prints the same style of command preview the WebUI backend would generate.

## Step 5: Launch SFT without clicking buttons

```powershell
.venv-py314/Scripts/python.exe scripts/ui_backdoor_train.py `
  --config examples/distillation/gemma4_student_sft_ui_backdoor_template.yaml `
  --print-training-args-path
```

Expected output artifacts:

1. SFT checkpoint directory under `saves/gemma4_student/lora/sft`.
2. `training_args.yaml` generated by the backend helper.

## Step 6: Launch DPO without clicking buttons

```powershell
.venv-py314/Scripts/python.exe scripts/ui_backdoor_train.py `
  --config examples/distillation/gemma4_student_dpo_ui_backdoor_template.yaml `
  --print-training-args-path
```

Expected output artifacts:

1. DPO checkpoint directory under `saves/gemma4_student/lora/dpo`.
2. `training_args.yaml` for the DPO run.

## Step 7: Merge SFT+DPO adapters

```powershell
.venv-py314/Scripts/python.exe -m llamafactory.cli export `
  examples/merge_lora/gemma4_student_merge_ui_backdoor_template.yaml
```

Expected merged model output:

1. `saves/gemma4_student/merged/four_lang_ocr_chat`

## Step 8: Evaluate policy and task quality

Minimum checks:

1. Translation accuracy for EN/RO/FR/HU.
2. OCR text fidelity against source text.
3. Chat coherence and refusal behavior for unsupported languages.

Suggested metrics:

1. Translation: BLEU, chrF, COMET.
2. OCR: CER, WER.
3. Chat policy: pass rate on language-gate test set.

Quick language-gate evaluator (already added):

1. Eval spec: [data/lang_gate_eval_4lang.jsonl](data/lang_gate_eval_4lang.jsonl)
2. Evaluator script: [scripts/eval_lang_gate_4lang.py](scripts/eval_lang_gate_4lang.py)

Example usage:

```powershell
.venv-py314/Scripts/python.exe scripts/eval_lang_gate_4lang.py `
   --spec data/lang_gate_eval_4lang.jsonl `
   --pred benchmark_output/lang_gate/predictions.jsonl `
   --out benchmark_output/lang_gate/report.json
```

## Requested 1-2-3 execution map

This maps directly to the three actions requested in chat:

1. Action 1 (start real SFT now):

```powershell
.venv-py314/Scripts/python.exe scripts/ui_backdoor_train.py `
   --config examples/distillation/gemma4_student_sft_ui_backdoor_template.yaml
```

2. Action 2 (define exact dataset schema):

1. SFT schema: `instruction`, `input`, `output`
2. DPO schema: `prompt`, `chosen`, `rejected`
3. Dataset registry entries:
    1. `distill_ocr_chat_translate_4lang_train`
    2. `distill_ocr_chat_translate_4lang_pref`

3. Action 3 (wire language-gate eval + quick evaluator):

1. Spec file with `id`, `prompt`, `input_language`, `should_answer`
2. Prediction file with `id`, `response`
3. Evaluator script reports `total`, `passed`, `score`

## Size and time expectations

Common outcomes for this exact specialization:

1. 9B teacher to 1.5B student:
   1. About 6x fewer parameters.
   2. 4-bit size usually around 0.9 GB to 1.4 GB for base text-only weights.
2. 9B teacher to 3B student:
   1. About 3x fewer parameters.
   2. 4-bit size usually around 1.8 GB to 2.5 GB for base text-only weights.

Wall-clock estimate on one strong consumer GPU:

1. SFT: 8 to 24 hours.
2. DPO: 4 to 12 hours.
3. Merge and checks: 1 to 4 hours.
4. First solid iteration total: around 1 to 3 days.

## Notes on strict 4-language behavior

Distillation alone cannot guarantee 100% language exclusion.

For stronger control:

1. Keep training data strictly within the four target languages.
2. Add many refusal examples for other languages.
3. Add a runtime language gate before generation.

## Teacher capability profiler in WebUI (new)

You can now profile one or many teacher models directly in the Train tab.

Where:

1. Open `Train` tab.
2. Expand `Teacher capability profiler`.

Inputs:

1. `Teacher models`: one model per line.
2. `Role overrides (optional)`: `model=role1,role2`.
3. `Capability focus`: choose target capabilities like translation, ocr, chat, policy.

Action:

1. Click `Analyze teachers`.

Outputs:

1. A summary panel with inferred capabilities and assigned roles.
2. A saved profile file path under `llamaboard_config/teacher_capability_profiles/`.

How to use for multi-teacher distillation:

1. Put your translation-focused teacher and OCR-focused teacher in the model list.
2. Assign explicit role overrides if needed.
3. Use the saved profile JSON as your planning artifact when curating SFT/DPO datasets.
