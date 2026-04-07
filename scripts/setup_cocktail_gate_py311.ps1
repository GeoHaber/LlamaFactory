Param(
    [switch]$CpuOnly
)

$ErrorActionPreference = "Stop"

Write-Host "setup_cocktail_gate_py311.ps1 is deprecated. Redirecting to Python 3.14 setup..." -ForegroundColor Yellow

$args = @()
if ($CpuOnly) {
    $args += "-CpuOnly"
}

& "$PSScriptRoot/setup_cocktail_gate_py314.ps1" @args
