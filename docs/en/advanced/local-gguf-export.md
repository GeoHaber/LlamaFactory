# Local GGUF Export Workflow (Windows, Python 3.14)

This workflow creates a local GGUF model from your LLaMA Factory output.

Pipeline:

1. Merge LoRA into base model (HF format)
2. Convert merged HF model to GGUF F16
3. Quantize GGUF (optional)

## Prerequisites

- Python 3.14 environment with LLaMA Factory installed
- A built llama.cpp checkout
  - `convert_hf_to_gguf.py` (or `convert-hf-to-gguf.py`)
  - `llama-quantize.exe` (or `quantize.exe`)

## One-command export

Use:

- [scripts/export_local_gguf.ps1](scripts/export_local_gguf.ps1)

Example:

```powershell
pwsh -File scripts/export_local_gguf.ps1 `
  -PythonExe "C:/Users/dvdze/AppData/Local/Python/pythoncore-3.14-64/python.exe" `
  -BaseModel "Qwen/Qwen2.5-Coder-14B-Instruct" `
  -AdapterPath "saves/qwen2.5-coder-14b/lora/distill_dpo" `
  -Template "qwen" `
  -ExportDir "saves/gguf_export/qwen25_merged_hf" `
  -LlamaCppDir "D:/src/llama.cpp" `
  -OutFile "saves/gguf_export/qwen25_distill_q4_k_m.gguf" `
  -Quantization "Q4_K_M"
```

## Quantization options

- `F16` (no quantization, full precision GGUF)
- `Q8_0`
- `Q6_K`
- `Q5_K_M`
- `Q4_K_M` (common local default)

## Re-quantize only

If you already have a merged HF export and only want GGUF conversion/quantization:

```powershell
pwsh -File scripts/export_local_gguf.ps1 `
  -PythonExe "C:/Users/dvdze/AppData/Local/Python/pythoncore-3.14-64/python.exe" `
  -BaseModel "Qwen/Qwen2.5-Coder-14B-Instruct" `
  -Template "qwen" `
  -ExportDir "saves/gguf_export/qwen25_merged_hf" `
  -LlamaCppDir "D:/src/llama.cpp" `
  -OutFile "saves/gguf_export/qwen25_distill_q5_k_m.gguf" `
  -Quantization "Q5_K_M" `
  -SkipMergeExport
```

## Notes

- If you get template mismatch errors during export, verify `-Template` matches your base model family.
- Keep the F16 GGUF artifact for future quantization variants.
- For AMD/Vulkan local inference, run the resulting `.gguf` with your preferred runner that supports Vulkan backends.
