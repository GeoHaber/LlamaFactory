# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
from pathlib import Path
from typing import Any, cast

import yaml  # xray: ignore[SEC-015]


def _as_dict(obj: Any) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError("Expected a JSON/YAML object.")
    return cast(dict[str, Any], obj)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data_any: Any = yaml.safe_load(f) or {}
    data = _as_dict(data_any)
    return dict(data)


def _read_dataset_info(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data_any: Any = json.load(f)  # xray: ignore[PY-005]
    data = _as_dict(data_any)
    return dict(data)


def _resolve_data_file(dataset_entry: dict[str, Any], dataset_dir: Path) -> Path | None:
    file_name = dataset_entry.get("file_name")
    if not isinstance(file_name, str) or not file_name:
        return None
    return dataset_dir / file_name


def _first_jsonl_record(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)  # xray: ignore[PY-005]
            if isinstance(row, dict):
                return cast(dict[str, Any], row)
            return None

    return None


def _check_model_template(cfg: dict[str, Any], errors: list[str], warnings: list[str]) -> None:
    model = str(cfg.get("model_name_or_path", "")).lower()
    template = str(cfg.get("template", "")).lower()
    if not model or not template:
        errors.append("Config missing model_name_or_path or template.")
        return

    if "qwen" in model and "qwen" not in template:
        errors.append(f"Model '{cfg.get('model_name_or_path')}' should use a qwen template, got '{cfg.get('template')}'.")

    if "gpt2" in model and "qwen" in template:
        errors.append(f"Model '{cfg.get('model_name_or_path')}' is incompatible with template '{cfg.get('template')}'.")

    if "llama" in model and "qwen" in template:
        warnings.append("LLaMA-family model with qwen template looks suspicious. Verify prompt template compatibility.")


def _check_lora_sanity(cfg: dict[str, Any], errors: list[str], warnings: list[str]) -> None:
    if str(cfg.get("finetuning_type", "")).lower() != "lora":
        return

    lora_target = str(cfg.get("lora_target", "")).strip()
    if not lora_target:
        errors.append("LoRA config missing lora_target.")
        return

    if lora_target.lower() == "all":
        return

    targets = [x.strip() for x in lora_target.split(",") if x.strip()]
    if not targets:
        errors.append("lora_target is empty after parsing.")
        return

    model = str(cfg.get("model_name_or_path", "")).lower()
    if "gpt2" in model and any(x in {"q_proj", "k_proj", "v_proj"} for x in targets):
        warnings.append("GPT-2 style model usually uses Conv1D targets (c_attn,c_proj,c_fc), not q_proj/k_proj/v_proj.")


def _check_dataset_schema(
    cfg: dict[str, Any], dataset_info: dict[str, Any], dataset_info_path: Path, errors: list[str], warnings: list[str]
) -> None:
    dataset_name = cfg.get("dataset")
    dataset_dir = cfg.get("dataset_dir", "data")
    if not dataset_name:
        errors.append("Config missing dataset.")
        return

    if dataset_name not in dataset_info:
        errors.append(f"Dataset '{dataset_name}' not found in {dataset_info_path}.")
        return

    entry_obj: Any = dataset_info[dataset_name]
    if not isinstance(entry_obj, dict):
        errors.append(f"Dataset entry for '{dataset_name}' must be an object.")
        return
    entry = _as_dict(entry_obj)

    entry_columns: Any = entry.get("columns", {})
    columns = _as_dict(entry_columns) if isinstance(entry_columns, dict) else {}
    ranking = bool(entry.get("ranking", False))

    data_file = _resolve_data_file(entry, Path(str(dataset_dir)))
    if data_file is None:
        warnings.append(f"Dataset '{dataset_name}' has no local file_name, skipping local schema probe.")
        return

    first_row = _first_jsonl_record(data_file)
    if first_row is None:
        warnings.append(f"Dataset file '{data_file}' missing or empty, skipping row-level schema checks.")
        return

    default_prompt = str(columns.get("prompt", "instruction"))
    if default_prompt not in first_row:
        if "prompt" in first_row and "prompt" not in columns:
            errors.append(
                "Dataset prompt key mismatch: row has 'prompt' but dataset_info columns.prompt is not set. "
                "Add columns.prompt='prompt'."
            )
        else:
            errors.append(f"Prompt column '{default_prompt}' not found in dataset row.")

    if ranking:
        chosen_key = str(columns.get("chosen", "chosen"))
        rejected_key = str(columns.get("rejected", "rejected"))
        if chosen_key not in first_row:
            errors.append(f"Ranking dataset missing chosen column '{chosen_key}' in row.")
        if rejected_key not in first_row:
            errors.append(f"Ranking dataset missing rejected column '{rejected_key}' in row.")


def run_preflight(config_path: Path, dataset_info_path: Path) -> int:
    cfg = _read_yaml(config_path)
    dataset_info = _read_dataset_info(dataset_info_path)

    errors: list[str] = []
    warnings: list[str] = []

    _check_model_template(cfg, errors, warnings)
    _check_lora_sanity(cfg, errors, warnings)
    _check_dataset_schema(cfg, dataset_info, dataset_info_path, errors, warnings)

    if warnings:
        print("[preflight] warnings:")  # xray: ignore[PY-004]
        for item in warnings:
            print(f"  - {item}")  # xray: ignore[PY-004]

    if errors:
        print("[preflight] errors:")  # xray: ignore[PY-004]
        for item in errors:
            print(f"  - {item}")  # xray: ignore[PY-004]
        return 1

    print(f"[preflight] OK: {config_path}")  # xray: ignore[PY-004]
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate distillation gate config before training.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-info", default="data/dataset_info.json")
    args = parser.parse_args()

    exit_code = run_preflight(Path(args.config), Path(args.dataset_info))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
