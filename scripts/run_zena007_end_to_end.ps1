# Zena007 End-to-End Multi-Teacher Distillation Pipeline
#
# Architecture (v4 -- SPSC Ring-Buffer FIFO Dispatch):
#   Per-teacher SPSCRingBuffer (lock-free, GIL-atomic) with 2048-slot depth
#   via --fifo-size 0 (auto mode). RAMPressureThrottle with hysteresis gates
#   workers when system memory is tight. Adaptive decoding budgets select
#   max_tokens/temperature per prompt category.
#
# Pipeline stages:
#   1. Generate multi-teacher responses (FIFO dispatch, checkpoint/resume)
#   2. Purify into consensus (SFT) and conflict (DPO) splits
#   3. Auto-generate training configs (SFT, [DPO,] merge YAML)
#   4. Register datasets in dataset_info.json (idempotent)
#   5. [Sequential] Train SFT -> [DPO if samples] -> Merge -> Smoke test
#      [Forge Matrix] Train N variants in parallel -> Eval panel -> Merge champion
#
# Crash-Resume:
#   Every stage is idempotent — just re-run the same command after a crash.
#   Stage              Skip condition                             Resume behaviour
#   ─────────────────  ─────────────────────────────────────────  ─────────────────────────────────────
#   Generation         teacher_responses.jsonl has expected rows  Per-teacher checkpoints merged; gaps filled
#   Purification       purification_report.json exists            Full skip
#   Config gen         *_sft.yaml and *_merge.yaml both exist     Full skip
#   Dataset reg        Entry already in dataset_info.json         Idempotent write
#   SFT/DPO training   adapter_model.safetensors present          Auto-resumes from latest checkpoint-N
#   DPO training       No conflict_dpo.jsonl samples              Skipped entirely (merged model uses SFT only)
#   Merge              saves/$Tag/merged/config.json exists       Full skip
#
# Config generation rules (gen_distill_configs.py):
#   - merge config uses SFT adapter only when no DPO data (_merge_config has_dpo=False)
#   - NEVER add keys unknown to HfArgumentParser (e.g. booster) — causes ValueError at startup
#   - dataset_info.json file_name is relative to dataset_dir ("data/") — no leading "data/" prefix
#
# Usage:
#   ./scripts/run_zena007_end_to_end.ps1                          # sequential path
#   ./scripts/run_zena007_end_to_end.ps1 -SkipTrain -SkipDpo     # generation + purify only
#   ./scripts/run_zena007_end_to_end.ps1 -SkipMerge              # skip merge step
#   ./scripts/run_zena007_end_to_end.ps1 -UseForge               # parallel Student Forge Matrix
#   ./scripts/run_zena007_end_to_end.ps1 -UseForge -SkipEval     # forge without eval panel
#   ./scripts/run_zena007_end_to_end.ps1 -ReExamModelId zena007/B  # re-exam a specific student

Param(
    [string]$StudentModel = "Qwen/Qwen2.5-1.5B-Instruct",
    [string]$Tag = "zena007",
    [switch]$SkipTrain,
    [switch]$SkipDpo,
    [switch]$SkipMerge,
    [switch]$SkipSmokeTest,
    # --- Student Forge Matrix switches ---
    [switch]$UseForge,
    [string]$ForgeMatrix = "data/forge_matrix/zena007_matrix.yaml",
    [switch]$SkipEval,
    [switch]$SkipSlimDown,
    [string]$ReExamModelId = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repoRoot

$py = ".venv-py314/Scripts/python.exe"
if (-not (Test-Path $py)) {
    throw "Expected interpreter not found: $py"
}

$responsesPath = "data/zena007/teacher_responses.jsonl"
$promptsPath = "data/zena007_prompts.jsonl"
$purifiedDir = "data/zena007/purified"
$configDir = "examples/distillation/auto"
$sftCfg = Join-Path $configDir "$Tag`_sft.yaml"
$dpoCfg = Join-Path $configDir "$Tag`_dpo.yaml"
$mergeCfg = Join-Path $configDir "$Tag`_merge.yaml"
$mergedDir = "saves/$Tag/merged"
$forgeResults = "saves/$Tag/forge_results.jsonl"
$championFile = "saves/$Tag/champion.txt"
$evalProbes = "data/zena007/purified/eval_probes.jsonl"

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

function Get-ConfigValue {
    param(
        [Parameter(Mandatory = $true)][string]$ConfigPath,
        [Parameter(Mandatory = $true)][string]$Key
    )

    $code = @"
import yaml
from pathlib import Path

cfg = yaml.safe_load(Path(r'''$ConfigPath''').read_text(encoding='utf-8')) or {}
val = cfg.get('$Key', '')
if val is None:
    val = ''
print(str(val))
"@
    $output = & $py -c $code
    return ($output | Select-Object -First 1)
}

function Get-LatestCheckpoint {
    param(
        [Parameter(Mandatory = $true)][string]$OutputDir
    )

    $resolved = Join-Path (Get-Location) $OutputDir
    if (-not (Test-Path $resolved)) {
        return ""
    }

    $latest = Get-ChildItem -Path $resolved -Directory -Filter "checkpoint-*" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^checkpoint-(\d+)$' } |
        Sort-Object { [int]($_.Name -replace '^checkpoint-', '') } -Descending |
        Select-Object -First 1

    return if ($null -eq $latest) { "" } else { $latest.FullName }
}

function Test-FileNonEmpty {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Test-Path $Path) -and ((Get-Item $Path).Length -gt 0)
}

function Test-TrainComplete {
    param([Parameter(Mandatory = $true)][string]$OutputDir)
    # LlamaFactory writes training_args.bin (or trainer_state.json in HF) when done.
    # We treat the run as complete when a model weight file is present alongside
    # the adapter config, meaning the run saved a final checkpoint.
    $resolved = Join-Path (Get-Location) $OutputDir
    return (Test-Path (Join-Path $resolved "adapter_config.json")) -and
           (Test-Path (Join-Path $resolved "adapter_model.safetensors"))
}

function Get-ExpectedPromptCount {
    param(
        [Parameter(Mandatory = $true)][string]$PromptFile
    )

    if (-not (Test-Path $PromptFile)) {
        throw "Prompts file not found: $PromptFile"
    }

    $code = @"
import json
from pathlib import Path

path = Path(r'''$PromptFile''')
raw = path.read_text(encoding='utf-8').strip()
if not raw:
    print(0)
    raise SystemExit(0)

if raw.startswith('['):
    rows = json.loads(raw)
    print(len(rows) if isinstance(rows, list) else 0)
else:
    print(sum(1 for line in raw.splitlines() if line.strip()))
"@

    $count = & $py -c $code
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to compute expected prompt count from $PromptFile"
    }
    return [int]($count | Select-Object -First 1)
}

function Wait-ForGenerationComplete {
    param(
        [Parameter(Mandatory = $true)][int]$ExpectedRows,
        [int]$TimeoutMinutes = 720,
        [int]$PollSeconds = 60,
        [int]$MaxStalePolls = 30
    )

    Write-Host "[wait] Waiting for teacher generation completion..."
    $deadline = (Get-Date).AddMinutes($TimeoutMinutes)
    $lastCount = -1
    $stalePolls = 0

    while ($true) {
        if ((Get-Date) -gt $deadline) {
            throw "Timed out waiting for teacher generation completion after $TimeoutMinutes minute(s)."
        }

        if (Test-Path $responsesPath) {
            $lineCount = (Get-Content $responsesPath | Measure-Object).Count
            if ($lineCount -ge $ExpectedRows) {
                Write-Host "[ok] Found $lineCount rows in $responsesPath"
                return
            }

            if ($lineCount -gt $lastCount) {
                $stalePolls = 0
            }
            else {
                $stalePolls += 1
            }

            $lockPath = "$responsesPath.lock"
            if (($stalePolls -ge $MaxStalePolls) -and (-not (Test-Path $lockPath))) {
                throw "Generation appears stalled at $lineCount/$ExpectedRows rows and lock file is missing: $lockPath"
            }

            $lastCount = $lineCount
            Write-Host "[wait] teacher_responses rows=$lineCount (need $ExpectedRows)"
        }
        else {
            Write-Host "[wait] teacher_responses not ready yet"
        }

        Start-Sleep -Seconds $PollSeconds
    }
}

function Update-DatasetInfo {
    Write-Host "[step] Register datasets in data/dataset_info.json"
    # Idempotent: only writes entries that are missing — safe to re-run after a crash.
    $code = @'
import json
from pathlib import Path

path = Path("data/dataset_info.json")
obj = json.loads(path.read_text(encoding="utf-8"))

changed = False

if "zena007_consensus_sft" not in obj:
    obj["zena007_consensus_sft"] = {
        "file_name": "data/zena007/purified/consensus_sft.jsonl",
        "columns": {"prompt": "instruction", "response": "output"}
    }
    changed = True

if "zena007_conflict_dpo" not in obj:
    obj["zena007_conflict_dpo"] = {
        "file_name": "data/zena007/purified/conflict_dpo.jsonl",
        "ranking": True,
        "columns": {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"}
    }
    changed = True

if changed:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("dataset_info.json updated: zena007_consensus_sft, zena007_conflict_dpo")
else:
    print("dataset_info.json already contains both zena007 entries — skipped.")
'@
    & $py -c $code
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to update data/dataset_info.json"
    }
}

function Start-GenerationIfNeeded {
    param(
        [Parameter(Mandatory = $true)][int]$ExpectedRows
    )

    if (Test-Path $responsesPath) {
        $rows = (Get-Content $responsesPath | Measure-Object).Count
        if ($rows -ge $ExpectedRows) {
            Write-Host "[skip] teacher responses already complete ($rows rows)"
            return
        }
    }

    $lockPath = "$responsesPath.lock"
    if (Test-Path $lockPath) {
        Write-Host "[info] generation lock found: $lockPath (assuming generator is already running)"
        return
    }

    Write-Host "[step] Starting generation with adaptive budgets"
    $genCmd = @(
        "scripts/multi_teacher_generate.py",
        "--manifest", "data/zena007/teacher_manifest.json",
        "--prompts", $promptsPath,
        "--out", $responsesPath,
        "--max-tokens", "512",
        "--temperature", "0.7",
        "--adaptive-budgets",
        "--dispatch-mode", "teacher-fifo",
        "--fifo-size", "0",
        "--ram-pause-pct", "12",
        "--ram-resume-pct", "22"
    )

    $p = Start-Process -FilePath $py -ArgumentList $genCmd -PassThru -WindowStyle Hidden
    Write-Host "[ok] generation started (pid=$($p.Id))"
}

$expectedPromptRows = Get-ExpectedPromptCount -PromptFile $promptsPath
if ($expectedPromptRows -le 0) {
    throw "Expected prompt count is zero from $promptsPath"
}

Write-Host "[info] Expected prompt rows: $expectedPromptRows"

Start-GenerationIfNeeded -ExpectedRows $expectedPromptRows
Wait-ForGenerationComplete -ExpectedRows $expectedPromptRows

$purifyReport = Join-Path $purifiedDir "purification_report.json"
if (Test-FileNonEmpty $purifyReport) {
    Write-Host "[skip] Purification already complete ($purifyReport exists)"
} else {
    Invoke-Step -Name "Purify teacher outputs" -Action {
        & $py scripts/purify_teacher_outputs.py --input $responsesPath --out-dir $purifiedDir --answer-threshold 0.85 --reason-threshold 0.6
    }
}

if ((Test-Path $sftCfg) -and (Test-Path $mergeCfg)) {
    Write-Host "[skip] Distillation configs already present in $configDir"
} else {
    Invoke-Step -Name "Generate distillation configs" -Action {
        & $py scripts/gen_distill_configs.py --student $StudentModel --data-dir $purifiedDir --out-dir $configDir --tag $Tag --cpu-safe
    }
}

Update-DatasetInfo

# ── Re-Exam shortcut: evaluate an existing student and exit ──────────────
if ($ReExamModelId -ne "") {
    Write-Host "[re-exam] Re-evaluating model: $ReExamModelId"
    Invoke-Step -Name "Re-Exam eval panel" -Action {
        & $py scripts/eval_student_panel.py --saves-tag $Tag --probes $evalProbes
    }
    Invoke-Step -Name "Update registry eval" -Action {
        & $py scripts/student_registry.py update-eval --model-id $ReExamModelId --eval-file "saves/$Tag/eval_scorecards.jsonl"
    }
    Invoke-Step -Name "Gap analysis" -Action {
        & $py scripts/student_registry.py gap-analysis --model-id $ReExamModelId
    }
    Write-Host "Done: re-exam for $ReExamModelId complete."
    exit 0
}

# ── Forge path vs sequential path ────────────────────────────────────────
if ($UseForge) {
    Write-Host "=== STUDENT FORGE MATRIX MODE ==="

    if (-not (Test-Path $ForgeMatrix)) {
        throw "Forge matrix not found: $ForgeMatrix"
    }

    Invoke-Step -Name "Run Student Forge (parallel training)" -Action {
        & $py scripts/run_student_forge.py --matrix $ForgeMatrix --tag $Tag
    }

    if (-not $SkipEval) {
        if (Test-Path $evalProbes) {
            Invoke-Step -Name "Eval Student Panel (two-pass verification)" -Action {
                & $py scripts/eval_student_panel.py --saves-tag $Tag --probes $evalProbes
            }
        }
        else {
            Write-Host "[skip] Eval probes not found: $evalProbes"
        }
    }
    else {
        Write-Host "[skip] Eval panel (disabled)"
    }

    # Register all successful variants
    if (Test-Path $forgeResults) {
        $regCode = @"
import json
from pathlib import Path

results = [json.loads(l) for l in Path('$($forgeResults -replace '\\','/')').read_text(encoding='utf-8').splitlines() if l.strip()]
ok = [r for r in results if r.get('ok')]
for r in ok:
    vid = r['variant_id']
    import subprocess, sys
    subprocess.run([sys.executable, 'scripts/student_registry.py', 'register', '--saves-tag', '$Tag', '--variant-id', vid])
"@
        & $py -c $regCode
    }

    # Merge champion adapter
    if ((-not $SkipMerge) -and (Test-Path $championFile)) {
        $champCode = @"
import json
from pathlib import Path

champ = json.loads(Path('$($championFile -replace '\\','/')').read_text(encoding='utf-8'))
vid = champ['variant_id']
model = champ['model']
adapter = champ['adapter_path']
print(f"CHAMPION={vid}")
print(f"MODEL={model}")
print(f"ADAPTER={adapter}")
"@
        $champInfo = & $py -c $champCode
        $champVid = ($champInfo | Where-Object { $_ -match "^CHAMPION=" }) -replace "^CHAMPION=", ""
        $champModel = ($champInfo | Where-Object { $_ -match "^MODEL=" }) -replace "^MODEL=", ""
        $champAdapter = ($champInfo | Where-Object { $_ -match "^ADAPTER=" }) -replace "^ADAPTER=", ""

        Write-Host "[merge] Champion: $champVid ($champModel + $champAdapter)"

        $mergeYaml = "saves/$Tag/merge_champion.yaml"
        $mergeCode = @"
from pathlib import Path
cfg = f'''model_name_or_path: {0}
adapter_name_or_path: {1}
template: qwen
finetuning_type: lora
export_dir: saves/{2}/merged
export_size: 2
export_legacy_format: false
'''.format('$champModel', '$champAdapter', '$Tag')
Path('$($mergeYaml -replace '\\','/')').parent.mkdir(parents=True, exist_ok=True)
Path('$($mergeYaml -replace '\\','/')').write_text(cfg, encoding='utf-8')
"@
        & $py -c $mergeCode
        Invoke-Step -Name "Merge champion adapter" -Action {
            & $py -m llamafactory.cli export $mergeYaml
        }
    }
    elseif ($SkipMerge) {
        Write-Host "[skip] Merge (disabled)"
    }
    else {
        Write-Host "[skip] Merge (no champion.txt found)"
    }

    # Slim-down (GGUF export + bench)
    if ((-not $SkipSlimDown) -and (Test-Path $championFile)) {
        Invoke-Step -Name "Slim-Down (GGUF export + speed bench)" -Action {
            $slimArgs = @("scripts/slim_down.py", "--saves-tag", $Tag, "--quants", "Q4_K_M", "Q5_K_M", "Q8_0")
            if (Test-Path $evalProbes) {
                $slimArgs += @("--probes", $evalProbes)
            }
            & $py @slimArgs
        }
    }
    elseif (-not $SkipSlimDown) {
        Write-Host "[skip] Slim-down (no champion.txt)"
    }
}
else {
    # ── Original sequential path — each stage checks its own completion   ─
    if (-not $SkipTrain) {
        $sftOutputDir = Get-ConfigValue -ConfigPath $sftCfg -Key "output_dir"
        if ((-not [string]::IsNullOrWhiteSpace($sftOutputDir)) -and (Test-TrainComplete $sftOutputDir)) {
            Write-Host "[skip] SFT training already complete ($sftOutputDir)"
        } else {
            $sftArgs = @("-m", "llamafactory.cli", "train", $sftCfg)
            if (-not [string]::IsNullOrWhiteSpace($sftOutputDir)) {
                $ckpt = Get-LatestCheckpoint -OutputDir $sftOutputDir
                if (-not [string]::IsNullOrWhiteSpace($ckpt)) {
                    Write-Host "[resume] SFT from checkpoint: $ckpt"
                    $sftArgs += @("--resume_from_checkpoint", $ckpt)
                }
            }
            Invoke-Step -Name "Train SFT" -Action {
                & $py @sftArgs
            }
        }
    }

    if ((-not $SkipDpo) -and (Test-FileNonEmpty "data/zena007/purified/conflict_dpo.jsonl")) {
        $dpoOutputDir = Get-ConfigValue -ConfigPath $dpoCfg -Key "output_dir"
        if ((-not [string]::IsNullOrWhiteSpace($dpoOutputDir)) -and (Test-TrainComplete $dpoOutputDir)) {
            Write-Host "[skip] DPO training already complete ($dpoOutputDir)"
        } else {
            $dpoArgs = @("-m", "llamafactory.cli", "train", $dpoCfg)
            if (-not [string]::IsNullOrWhiteSpace($dpoOutputDir)) {
                $ckpt = Get-LatestCheckpoint -OutputDir $dpoOutputDir
                if (-not [string]::IsNullOrWhiteSpace($ckpt)) {
                    Write-Host "[resume] DPO from checkpoint: $ckpt"
                    $dpoArgs += @("--resume_from_checkpoint", $ckpt)
                }
            }
            Invoke-Step -Name "Train DPO" -Action {
                & $py @dpoArgs
            }
        }
    } else {
        Write-Host "[skip] DPO (disabled or no conflict_dpo samples)"
    }

    if (-not $SkipMerge) {
        $mergeModelFile = Join-Path $mergedDir "config.json"
        if (Test-Path $mergeModelFile) {
            Write-Host "[skip] Merge already complete ($mergedDir)"
        } else {
            Invoke-Step -Name "Merge adapters" -Action {
                & $py -m llamafactory.cli export $mergeCfg
            }
        }
    }
}

if ((-not $SkipSmokeTest) -and (Test-Path $mergedDir)) {
    Write-Host "[step] Smoke test merged model"
    $smokeCode = @'
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "saves/zena007/merged"
tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

prompt = "Translate to Romanian: I hope your day is going great."
inputs = tok(prompt, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=40)
text = tok.decode(out[0], skip_special_tokens=True)
print("SMOKE_OUTPUT:")
print(text)
'@
    & $py -c $smokeCode
    if ($LASTEXITCODE -ne 0) {
        throw "Smoke test failed"
    }
}

Write-Host "Done: Zena_007 end-to-end pipeline completed$(if ($UseForge) { ' (Forge Matrix mode)' } else { '' })."
