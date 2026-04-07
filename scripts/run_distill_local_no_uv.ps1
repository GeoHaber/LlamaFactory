Param(
    [switch]$RunSft,
    [switch]$RunDpo,
    [switch]$RunMerge,
    [switch]$RunEval,
    [switch]$RunAll
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repoRoot

$py = ".venv-py314/Scripts/python.exe"
if (-not (Test-Path $py)) {
    throw "Expected interpreter not found: $py"
}

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$Action
    )

    Write-Host "[step] $Name"
    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Name (exit code $LASTEXITCODE)"
    }
}

if ($RunAll) {
    $RunSft = $true
    $RunDpo = $true
    $RunMerge = $true
    $RunEval = $true
}

if (-not ($RunSft -or $RunDpo -or $RunMerge -or $RunEval)) {
    Write-Host "No stage selected. Use -RunAll or one or more stage switches."
    exit 1
}

if ($RunSft) {
    Invoke-Step -Name "SFT local" -Action {
        & $py scripts/ui_backdoor_train.py --config examples/distillation/gemma4_student_sft_ui_backdoor_local_run.yaml --print-training-args-path
    }
}

if ($RunDpo) {
    Invoke-Step -Name "DPO local" -Action {
        & $py scripts/ui_backdoor_train.py --config examples/distillation/gemma4_student_dpo_ui_backdoor_local_run.yaml --print-training-args-path
    }
}

if ($RunMerge) {
    Invoke-Step -Name "Merge local" -Action {
        & $py -m llamafactory.cli export examples/merge_lora/gemma4_student_merge_ui_backdoor_local_run.yaml
    }
}

if ($RunEval) {
    Invoke-Step -Name "Generate guarded predictions" -Action {
        & $py scripts/gen_lang_gate_predictions_local.py --model saves/gemma4_student/merged/four_lang_ocr_chat_local --spec data/lang_gate_eval_4lang.jsonl --out benchmark_output/lang_gate/predictions_local_merged_policy_guarded.jsonl --max-new-tokens 72 --policy-prefix "You are a specialized assistant. You can only support English, Romanian, French, and Hungarian. For any other language request, answer exactly: I can only support English, Romanian, French, and Hungarian. Please choose one of these languages." --guard-language-policy
    }

    Invoke-Step -Name "Evaluate guarded predictions" -Action {
        & $py scripts/eval_lang_gate_4lang.py --spec data/lang_gate_eval_4lang.jsonl --pred benchmark_output/lang_gate/predictions_local_merged_policy_guarded.jsonl --out benchmark_output/lang_gate/report_local_merged_policy_guarded.json
    }
}

Write-Host "Done."