Param(
    [Parameter(Mandatory = $true)][string]$PythonExe,
    [Parameter(Mandatory = $true)][string]$BaseModel,
    [string]$AdapterPath = "",
    [Parameter(Mandatory = $true)][string]$Template,
    [string]$ExportDir = "saves/gguf_export/merged_hf",
    [Parameter(Mandatory = $true)][string]$LlamaCppDir,
    [string]$OutFile = "saves/gguf_export/model.gguf",
    [ValidateSet("F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M")][string]$Quantization = "Q4_K_M",
    [switch]$SkipMergeExport
)

$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot/.."
$env:PYTHONPATH = "src"

function Resolve-ConvertScript {
    param([Parameter(Mandatory = $true)][string]$Root)

    $candA = Join-Path $Root "convert_hf_to_gguf.py"
    $candB = Join-Path $Root "convert-hf-to-gguf.py"
    if (Test-Path $candA) { return $candA }
    if (Test-Path $candB) { return $candB }
    throw "Cannot find convert_hf_to_gguf.py or convert-hf-to-gguf.py under $Root"
}

function Resolve-QuantizeExe {
    param([Parameter(Mandatory = $true)][string]$Root)

    $candA = Join-Path $Root "build/bin/llama-quantize.exe"
    $candB = Join-Path $Root "build/bin/quantize.exe"
    $candC = Join-Path $Root "llama-quantize.exe"
    if (Test-Path $candA) { return $candA }
    if (Test-Path $candB) { return $candB }
    if (Test-Path $candC) { return $candC }
    throw "Cannot find llama-quantize executable under $Root"
}

function New-ExportConfig {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Model,
        [Parameter(Mandatory = $true)][string]$TemplateName,
        [Parameter(Mandatory = $true)][string]$ExportPath,
        [string]$Adapter = ""
    )

    $lines = @(
        "### model",
        "model_name_or_path: $Model",
        "template: $TemplateName",
        "trust_remote_code: true",
        "",
        "### export",
        "export_dir: $ExportPath",
        "export_size: 5",
        "export_device: cpu",
        "export_legacy_format: false"
    )

    if (-not [string]::IsNullOrWhiteSpace($Adapter)) {
        $lines = @(
            "### model",
            "model_name_or_path: $Model",
            "adapter_name_or_path: $Adapter",
            "template: $TemplateName",
            "trust_remote_code: true",
            "",
            "### export",
            "export_dir: $ExportPath",
            "export_size: 5",
            "export_device: cpu",
            "export_legacy_format: false"
        )
    }

    $dir = Split-Path -Parent $Path
    if (-not (Test-Path $dir)) {
        New-Item -Path $dir -ItemType Directory -Force | Out-Null
    }

    Set-Content -Path $Path -Value ($lines -join "`n") -Encoding UTF8
}

$convertScript = Resolve-ConvertScript -Root $LlamaCppDir
$f16Out = [System.IO.Path]::ChangeExtension($OutFile, ".f16.gguf")

if (-not $SkipMergeExport) {
    $tempConfig = Join-Path $env:TEMP ("lf_export_gguf_{0}.yaml" -f [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds())
    Write-Host "Creating temporary export config: $tempConfig" -ForegroundColor Cyan
    New-ExportConfig -Path $tempConfig -Model $BaseModel -TemplateName $Template -ExportPath $ExportDir -Adapter $AdapterPath

    Write-Host "Exporting merged HF model with LLaMA Factory..." -ForegroundColor Cyan
    & $PythonExe -m llamafactory.cli export $tempConfig
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
    Write-Host "Skipping merge/export step (--SkipMergeExport)." -ForegroundColor Yellow
}

Write-Host "Converting HF model to GGUF (F16)..." -ForegroundColor Cyan
& $PythonExe $convertScript $ExportDir --outfile $f16Out --outtype f16
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Quantization -eq "F16") {
    Copy-Item -Path $f16Out -Destination $OutFile -Force
    Write-Host "Done: $OutFile" -ForegroundColor Green
    exit 0
}

$quantizeExe = Resolve-QuantizeExe -Root $LlamaCppDir
Write-Host "Quantizing GGUF to $Quantization..." -ForegroundColor Cyan
& $quantizeExe $f16Out $OutFile $Quantization
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Done: $OutFile" -ForegroundColor Green
