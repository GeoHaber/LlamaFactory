# Troubleshooting

Common errors encountered during LlamaFactory distillation pipeline usage.

---

## 1. `ValueError: Some keys are not used` at training startup

**Cause**: Extra keys in YAML config that `HfArgumentParser` doesn't recognize
(e.g., `booster: auto`, `custom_key: value`).

**Fix**: Remove the key from your YAML config. Only standard LlamaFactory
hyperparameter keys are allowed. See `src/llamafactory/hparams/` dataclasses
for valid keys.

---

## 2. `FileNotFoundError: data/data/zena007/...`

**Cause**: `dataset_info.json` `file_name` has a `data/` prefix, but
LlamaFactory auto-prepends `dataset_dir` (default `"data"`), creating a
double-path like `data/data/zena007/...`.

**Fix**: Strip the leading `data/` from `file_name` in `dataset_info.json`.
Use the `_rel_to_data(path)` helper in `run_student_forge.py`.

---

## 3. `ModuleNotFoundError: No module named 'zen_core_libs'`

**Cause**: `zen_core_libs` is not installed in the active Python environment.

**Fix**:
```bash
pip install -e /path/to/zen_core_libs
```

---

## 4. RAM-pressure throttle never resumes (stuck workers)

**Cause**: `psutil` not installed — the throttle runs as a no-op but doesn't
print a warning.

**Fix**: `pip install psutil`. Verify with:
```python
from zen_core_libs.common.system import RAMPressureThrottle
t = RAMPressureThrottle(); t.start()
print(t.stats())  # should show available_pct < 100
t.stop()
```

---

## 5. GGUF export fails with `convert_hf_to_gguf.py not found`

**Cause**: `--llama-cpp-path` points to wrong directory, or llama.cpp not built.

**Fix**: Ensure the path contains `convert_hf_to_gguf.py`. If using fallback
mode, omit `--llama-cpp-path` and LlamaFactory's export CLI will be used.

---

## 6. `SPSCRingBuffer` full — prompts dropped silently

**Cause**: Consumer thread (collector) is slower than producers. Default FIFO
depth may be insufficient for many teachers.

**Fix**: Use `--fifo-size 0` (auto mode, default 2048 slots) or increase
manually with `--fifo-size 4096`.

---

## 7. Forge auto-heal loops indefinitely

**Cause**: A training variant fails repeatedly with the same error, and
`_diagnose_with_llm()` suggests the same fix each time.

**Fix**: Check `saves/<tag>/forge_state.json` for the `failed_variants` list.
Manually remove the failing variant from the matrix YAML and re-run.
