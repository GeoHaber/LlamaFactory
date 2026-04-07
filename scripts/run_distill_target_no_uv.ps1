Param(
    [switch]$RunSft,
    [switch]$RunDpo,
    [switch]$RunMerge,
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
}

if (-not ($RunSft -or $RunDpo -or $RunMerge)) {
    Write-Host "No stage selected. Use -RunAll or one or more stage switches."
    exit 1
}

if ($RunSft) {
    Invoke-Step -Name "SFT target" -Action {
        & $py scripts/ui_backdoor_train.py --config examples/distillation/gemma4_student_sft_ui_backdoor_template.yaml --print-training-args-path
    }
}

if ($RunDpo) {
    Invoke-Step -Name "DPO target" -Action {
        & $py scripts/ui_backdoor_train.py --config examples/distillation/gemma4_student_dpo_ui_backdoor_template.yaml --print-training-args-path
    }
}

if ($RunMerge) {
    Invoke-Step -Name "Merge target" -Action {
        & $py -m llamafactory.cli export examples/merge_lora/gemma4_student_merge_ui_backdoor_template.yaml
    }
}

Write-Host "Done."
