Param(
    [string]$PythonExe = "python",
    [string]$ApiBase = "http://127.0.0.1:8000/v1",
    [string]$ModelName = "Qwen/Qwen2.5-Coder-14B-Instruct",
    [string]$SftConfig = "examples/distillation/student_sft_distill.yaml",
    [string]$DpoConfig = "examples/distillation/student_dpo_distill.yaml",
    [string]$SmokeSftConfig = "examples/distillation/student_sft_distill_cpu_smoke.yaml",
    [string]$SmokeDpoConfig = "examples/distillation/student_dpo_distill_cpu_smoke.yaml",
    [int]$RetryCount = 2,
    [int]$RetryDelaySeconds = 10,
    [int]$ApiWaitSeconds = 60,
    [switch]$SkipTrain,
    [switch]$NoApiCheck,
    [switch]$NoPreflight,
    [switch]$RequirePython314,
    [switch]$EnableSelfHeal,
    [string]$SelfHealApiBase = "",
    [string]$SelfHealModel = "",
    [string]$EndpointMode = "auto",
    [string]$DatasetInfoPath = "data/dataset_info.json",
    [string]$LogDir = "benchmark_output/cocktail_gate_logs"
)

$ErrorActionPreference = "Stop"

Set-Location "$PSScriptRoot/.."
$env:PYTHONPATH = "src"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Invoke-PythonChecked {
  param(
    [Parameter(Mandatory = $true)][string[]]$Args,
    [string]$LogPath = ""
  )

  if ([string]::IsNullOrWhiteSpace($LogPath)) {
    & $PythonExe @Args
  } else {
    $null = & $PythonExe @Args 2>&1 | Tee-Object -FilePath $LogPath
  }

  return [int]$LASTEXITCODE
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
  $output = & $PythonExe -c $code
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

  $checkpoints = Get-ChildItem -Path $resolved -Directory -Filter "checkpoint-*" -ErrorAction SilentlyContinue
  if (-not $checkpoints) {
    return ""
  }

  $latest = $checkpoints |
    Where-Object { $_.Name -match '^checkpoint-(\d+)$' } |
    Sort-Object { [int]($_.Name -replace '^checkpoint-', '') } -Descending |
    Select-Object -First 1

  if ($null -eq $latest) {
    return ""
  }

  return $latest.FullName
}

function Invoke-Preflight {
  param(
    [Parameter(Mandatory = $true)][string]$ConfigPath
  )

  if ($NoPreflight) {
    Write-Host "Skipping config preflight (--NoPreflight)." -ForegroundColor Yellow
    return
  }

  Write-Host "Running preflight for $ConfigPath" -ForegroundColor Cyan
  $exitCode = Invoke-PythonChecked -Args @(
    "scripts/gate_preflight.py",
    "--config", $ConfigPath,
    "--dataset-info", $DatasetInfoPath
  )
  if ($exitCode -ne 0) {
    throw "Preflight failed for $ConfigPath"
  }
}

function Wait-ApiReady {
  if ($NoApiCheck) {
    Write-Host "Skipping API readiness check." -ForegroundColor Yellow
    return
  }

  $deadline = (Get-Date).AddSeconds($ApiWaitSeconds)
  $probeUrl = "$ApiBase/models"
  while ((Get-Date) -lt $deadline) {
    try {
      $null = Invoke-RestMethod -Uri $probeUrl -Method Get -TimeoutSec 10
      Write-Host "API is reachable at $probeUrl" -ForegroundColor Green
      return
    } catch {
      Write-Host "Waiting for API endpoint: $probeUrl" -ForegroundColor DarkYellow
      Start-Sleep -Seconds 3
    }
  }

  throw "API endpoint is not reachable at $probeUrl after $ApiWaitSeconds seconds."
}

function Check-PythonVersion {
  if (-not $RequirePython314) {
    return
  }

  $checkCode = "import sys; raise SystemExit(0 if sys.version_info[:2] == (3,14) else 1)"
  $exitCode = Invoke-PythonChecked -Args @("-c", $checkCode)
  if ($exitCode -ne 0) {
    throw "Python 3.14 is required. Current interpreter does not report version 3.14."
  }
}

function Resolve-EndpointMode {
  if ($EndpointMode -ne "auto") {
    Write-Host "Using explicit endpoint mode: $EndpointMode" -ForegroundColor DarkYellow
    return $EndpointMode
  }

  $probeArgs = @(
    "scripts/probe_openai_endpoint_mode.py",
    "--api-base", $ApiBase,
    "--model", $ModelName,
    "--timeout", "10"
  )
  if (-not [string]::IsNullOrWhiteSpace($SelfHealModel)) {
    # No API key flag is passed by default here; keep probing lightweight.
  }

  Write-Host "Probing endpoint compatibility mode..." -ForegroundColor Cyan
  $probeOutput = & $PythonExe @probeArgs
  if ($LASTEXITCODE -ne 0) {
    Write-Host "Endpoint mode probe failed, defaulting to chat mode." -ForegroundColor Yellow
    return "chat"
  }

  try {
    $payload = ($probeOutput | Select-Object -Last 1) | ConvertFrom-Json
    if ($payload.mode -in @("chat", "completions")) {
      Write-Host "Resolved endpoint mode: $($payload.mode)" -ForegroundColor Green
      return [string]$payload.mode
    }
  } catch {
    Write-Host "Endpoint probe output parse failed, defaulting to chat mode." -ForegroundColor Yellow
  }

  return "chat"
}

function Invoke-SelfHealAdvisor {
  param(
    [Parameter(Mandatory = $true)][string]$StepName,
    [Parameter(Mandatory = $true)][string]$LogPath
  )

  if (-not $EnableSelfHeal) {
    return
  }

  $advisorArgs = @("scripts/gate_self_heal_advisor.py", "--step", $StepName, "--log", $LogPath)
  if (-not [string]::IsNullOrWhiteSpace($SelfHealApiBase)) {
    $advisorArgs += @("--api-base", $SelfHealApiBase)
  }
  if (-not [string]::IsNullOrWhiteSpace($SelfHealModel)) {
    $advisorArgs += @("--model", $SelfHealModel)
  }

  Write-Host "Running self-heal advisor for step: $StepName" -ForegroundColor Yellow
  $null = Invoke-PythonChecked -Args $advisorArgs
}

function Get-FailureCategory {
  param(
    [Parameter(Mandatory = $true)][string]$LogPath
  )

  $text = Get-Content -Path $LogPath -Raw -ErrorAction SilentlyContinue
  if ([string]::IsNullOrWhiteSpace($text)) {
    return "generic"
  }

  if ($text -match "ConnectionError|ConnectTimeout|ReadTimeout|Failed to establish a new connection|HTTPError|404 Client Error|503 Service Unavailable") {
    return "network"
  }

  if ($text -match 'KeyError: ''instruction''|KeyError: "instruction"|JSONDecodeError|dataset') {
    return "schema"
  }

  if ($text -match "CUDA out of memory|out of memory") {
    return "oom"
  }

  if ($text -match "No `target_modules` passed|target_modules") {
    return "lora"
  }

  return "generic"
}

function Invoke-FixerLoop {
  param(
    [Parameter(Mandatory = $true)][string]$StepName,
    [Parameter(Mandatory = $true)][string]$LogPath,
    [string]$ConfigPath = ""
  )

  if (-not $EnableSelfHeal) {
    return
  }

  $planPath = Join-Path $LogDir ("{0}_plan.json" -f ($StepName -replace "\s+", "_"))
  $args = @(
    "scripts/gate_fixer_loop.py",
    "--step", $StepName,
    "--log", $LogPath,
    "--dataset-info", $DatasetInfoPath,
    "--plan-out", $planPath,
    "--apply"
  )

  if (-not [string]::IsNullOrWhiteSpace($ConfigPath)) {
    $args += @("--config", $ConfigPath)
  }
  if (-not [string]::IsNullOrWhiteSpace($SelfHealApiBase)) {
    $args += @("--api-base", $SelfHealApiBase)
  }
  if (-not [string]::IsNullOrWhiteSpace($SelfHealModel)) {
    $args += @("--model", $SelfHealModel)
  }

  Write-Host "Running fixer loop for step: $StepName" -ForegroundColor Yellow
  $null = Invoke-PythonChecked -Args $args

  Invoke-SelfHealAdvisor -StepName $StepName -LogPath $LogPath
}

function Build-TrainCommandArgs {
  param(
    [Parameter(Mandatory = $true)][string]$ConfigPath,
    [switch]$AllowResume
  )

  $args = @("-m", "llamafactory.cli", "train", $ConfigPath)
  if ($AllowResume) {
    $outputDir = Get-ConfigValue -ConfigPath $ConfigPath -Key "output_dir"
    if (-not [string]::IsNullOrWhiteSpace($outputDir)) {
      $latestCheckpoint = Get-LatestCheckpoint -OutputDir $outputDir
      if (-not [string]::IsNullOrWhiteSpace($latestCheckpoint)) {
        Write-Host "Resuming from checkpoint: $latestCheckpoint" -ForegroundColor DarkYellow
        $args += @("--resume_from_checkpoint", $latestCheckpoint)
      }
    }
  }

  return $args
}

function Invoke-Step {
  param(
    [Parameter(Mandatory = $true)][string]$StepName,
  [Parameter(Mandatory = $true)][string[]]$CommandArgs,
  [string]$ConfigPath = ""
  )

  $maxAttempts = $RetryCount + 1
  $delaySeconds = $RetryDelaySeconds
  $oomFixApplied = $false

  for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
    $logPath = Join-Path $LogDir ("{0}_{1}.log" -f ($StepName -replace "\s+", "_"), $attempt)
  Write-Host ("[{0}] {1} (attempt {2}/{3})" -f $StepName, ($CommandArgs -join " "), $attempt, $maxAttempts) -ForegroundColor Cyan
    $exitCode = Invoke-PythonChecked -Args $CommandArgs -LogPath $logPath
    if ($exitCode -eq 0) {
      Write-Host "Step succeeded: $StepName" -ForegroundColor Green
      return
    }

    Write-Host "Step failed: $StepName (exit $exitCode). Log: $logPath" -ForegroundColor Red
  $category = Get-FailureCategory -LogPath $logPath

  if ($category -eq "network") {
    $maxAttempts = [Math]::Max($maxAttempts, 4)
    $delaySeconds = [Math]::Max($delaySeconds, 30)
    Write-Host "Adaptive retry profile: network (longer backoff)." -ForegroundColor DarkYellow
  } elseif ($category -eq "schema") {
    $maxAttempts = $attempt
    Write-Host "Adaptive retry profile: schema/data error (fail fast)." -ForegroundColor DarkYellow
  } elseif ($category -eq "oom") {
    Write-Host "Adaptive retry profile: OOM (attempting one safe config fix)." -ForegroundColor DarkYellow
    if (-not $oomFixApplied) {
      Invoke-FixerLoop -StepName $StepName -LogPath $logPath -ConfigPath $ConfigPath
      $oomFixApplied = $true
      if (-not [string]::IsNullOrWhiteSpace($ConfigPath)) {
        $CommandArgs = Build-TrainCommandArgs -ConfigPath $ConfigPath -AllowResume
      }
      $maxAttempts = [Math]::Max($maxAttempts, $attempt + 1)
    } else {
      $maxAttempts = $attempt
    }
  } else {
    Invoke-FixerLoop -StepName $StepName -LogPath $logPath -ConfigPath $ConfigPath
  }

  if ($attempt -lt $maxAttempts) {
    Write-Host "Retrying in $delaySeconds second(s)..." -ForegroundColor DarkYellow
    Start-Sleep -Seconds $delaySeconds
    } else {
      throw "Step failed after retries: $StepName"
    }
  }
}

Check-PythonVersion
Wait-ApiReady
$resolvedEndpointMode = Resolve-EndpointMode

if (-not $SkipTrain) {
  Write-Host "Running shadow smoke gate before full distillation." -ForegroundColor Cyan
  Invoke-Preflight -ConfigPath $SmokeSftConfig
  Invoke-Preflight -ConfigPath $SmokeDpoConfig
  Invoke-Step -StepName "Smoke SFT distillation" -CommandArgs (Build-TrainCommandArgs -ConfigPath $SmokeSftConfig)
  Invoke-Step -StepName "Smoke DPO distillation" -CommandArgs (Build-TrainCommandArgs -ConfigPath $SmokeDpoConfig)

  Invoke-Preflight -ConfigPath $SftConfig
  Invoke-Preflight -ConfigPath $DpoConfig
  Invoke-Step -StepName "SFT distillation" -CommandArgs (Build-TrainCommandArgs -ConfigPath $SftConfig -AllowResume) -ConfigPath $SftConfig
  Invoke-Step -StepName "DPO distillation" -CommandArgs (Build-TrainCommandArgs -ConfigPath $DpoConfig -AllowResume) -ConfigPath $DpoConfig
} else {
  Write-Host "Skipping distillation steps (--SkipTrain)." -ForegroundColor Yellow
}

Invoke-Step -StepName "Baseline benchmark" -CommandArgs @(
  "scripts/benchmark_ad_coordinator.py",
  "--dataset", "data/python_rust_bench_demo.jsonl",
  "--api_base", $ApiBase,
  "--model", $ModelName,
  "--endpoint_mode", $resolvedEndpointMode,
  "--compare_policies",
  "--output_dir", "benchmark_output/cocktail_gate_baseline"
)

Invoke-Step -StepName "Coordinator benchmark" -CommandArgs @(
  "scripts/benchmark_ad_coordinator.py",
  "--dataset", "data/python_rust_bench_demo.jsonl",
  "--api_base", $ApiBase,
  "--model", $ModelName,
  "--endpoint_mode", $resolvedEndpointMode,
  "--use_ad_coordinator",
  "--compare_policies",
  "--output_dir", "benchmark_output/cocktail_gate_coordinator"
)

Write-Host "Cocktail gate completed." -ForegroundColor Green
