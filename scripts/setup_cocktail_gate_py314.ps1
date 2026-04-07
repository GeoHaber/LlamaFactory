Param(
    [string]$VenvPath = ".venv-py314",
    [string]$Python314 = "py -3.14",
    [string]$TorchVersion = "2.11.0",
    [string]$TorchVisionVersion = "0.26.0",
    [string]$TorchAudioVersion = "2.11.0",
    [string]$DatasetsVersion = "4.8.4",
    [string]$DillVersion = "0.4.1",
    [string]$MultiprocessVersion = "0.70.19",
    [string]$TransformersVersion = "4.57.6",
    [switch]$CpuOnly
)

$ErrorActionPreference = "Stop"

Set-Location "$PSScriptRoot/.."

Write-Host "[1/7] Creating Python 3.14 virtual environment at $VenvPath" -ForegroundColor Cyan
Invoke-Expression "$Python314 -m venv $VenvPath"

$py = Join-Path $VenvPath "Scripts/python.exe"
if (!(Test-Path $py)) {
    throw "Python executable not found in venv: $py"
}

Write-Host "[2/7] Upgrading pip/wheel and pinning setuptools for torch compatibility" -ForegroundColor Cyan
& $py -m pip install --upgrade pip wheel "setuptools<82"

Write-Host "[3/7] Installing torch stack" -ForegroundColor Cyan
if ($CpuOnly) {
    & $py -m pip install torch==$TorchVersion torchvision==$TorchVisionVersion torchaudio==$TorchAudioVersion --index-url https://download.pytorch.org/whl/cpu
} else {
    # CUDA 12.8 wheels for modern NVIDIA drivers
    & $py -m pip install torch==$TorchVersion torchvision==$TorchVisionVersion torchaudio==$TorchAudioVersion --index-url https://download.pytorch.org/whl/cu128
}

Write-Host "[4/7] Installing LlamaFactory" -ForegroundColor Cyan
& $py -m pip install -e .

Write-Host "[5/7] Installing Python 3.14-compatible data stack" -ForegroundColor Cyan
& $py -m pip install datasets==$DatasetsVersion dill==$DillVersion multiprocess==$MultiprocessVersion

Write-Host "[5.5/7] Pinning transformers to stable 3.14-compatible release" -ForegroundColor Cyan
& $py -m pip install transformers==$TransformersVersion

Write-Host "[6/7] Installing gate runtime deps" -ForegroundColor Cyan
& $py -m pip install "accelerate>=1.3.0,<=1.11.0" "trl>=0.18.0,<=0.24.0" "peft>=0.18.0,<=0.18.1" requests

Write-Host "[7/7] Verifying environment" -ForegroundColor Cyan
& $py -c "import sys, torch, torchaudio, transformers, datasets, dill, multiprocess; print('python=', sys.version); print('torch=', torch.__version__); print('torchaudio=', torchaudio.__version__); print('cuda_available=', torch.cuda.is_available()); print('cuda_count=', torch.cuda.device_count()); print('transformers=', transformers.__version__); print('datasets=', datasets.__version__); print('dill=', dill.__version__); print('multiprocess=', multiprocess.__version__)"

Write-Host "Environment ready. Use: $py" -ForegroundColor Green
