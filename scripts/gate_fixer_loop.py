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
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import requests  # xray: ignore[SEC-015]
import yaml  # xray: ignore[SEC-015]


_ALLOWED_ACTIONS = {"set_yaml_key", "ensure_dataset_columns"}
_ALLOWED_YAML_KEYS = {
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "lora_target",
    "preprocessing_num_workers",
    "dataloader_num_workers",
}
_ALLOWED_COLUMNS = {"prompt", "query", "response", "chosen", "rejected", "messages"}


def _read_log(path: Path, max_chars: int = 15000) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)


def _read_dataset_info(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)  # xray: ignore[PY-005]
    if not isinstance(data, dict):
        raise ValueError(f"Dataset info must be object: {path}")
    return data


def _write_dataset_info(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)
        f.write("\n")


def _first_jsonl_record(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)  # xray: ignore[PY-005]
            if isinstance(obj, dict):
                return obj
            return None
    return None


def _categorize_failure(log_text: str) -> str:
    if re.search(
        r"ConnectionError|ConnectTimeout|ReadTimeout|Failed to establish a new connection|HTTPError|404 Client Error|503 Service Unavailable",
        log_text,
        re.IGNORECASE,
    ):
        return "network"
    if re.search(r"KeyError: 'instruction'|KeyError: \"instruction\"|JSONDecodeError|ValueError:.*dataset", log_text, re.IGNORECASE):
        return "schema"
    if re.search(r"CUDA out of memory|out of memory", log_text, re.IGNORECASE):
        return "oom"
    if re.search(r"No `target_modules` passed|target_modules", log_text, re.IGNORECASE):
        return "lora"
    return "generic"


def _validate_action(action: dict[str, Any]) -> bool:
    action_type = action.get("type")
    if action_type not in _ALLOWED_ACTIONS:
        return False

    if action_type == "set_yaml_key":
        return action.get("key") in _ALLOWED_YAML_KEYS

    if action_type == "ensure_dataset_columns":
        cols = action.get("columns", {})
        return isinstance(cols, dict) and set(cols.keys()).issubset(_ALLOWED_COLUMNS)

    return False


def _rule_based_plan(
    *,
    step: str,
    log_text: str,
    config_path: Path | None,
    dataset_info_path: Path,
) -> dict[str, Any]:
    plan: dict[str, Any] = {"reason": [], "actions": []}
    category = _categorize_failure(log_text)
    plan["reason"].append(f"failure_category={category}")

    cfg: dict[str, Any] = {}
    if config_path and config_path.exists():
        cfg = _read_yaml(config_path)

    if category == "oom" and cfg:
        batch = int(cfg.get("per_device_train_batch_size", 1))
        grad = int(cfg.get("gradient_accumulation_steps", 1))
        if batch > 1:
            new_batch = max(1, batch // 2)
            scale = max(1, batch // new_batch)
            new_grad = grad * scale
            plan["actions"].append({"type": "set_yaml_key", "key": "per_device_train_batch_size", "value": new_batch})
            plan["actions"].append({"type": "set_yaml_key", "key": "gradient_accumulation_steps", "value": new_grad})
            plan["reason"].append("auto-lowered batch size and compensated grad accumulation")
        elif grad < 64:
            plan["actions"].append({"type": "set_yaml_key", "key": "gradient_accumulation_steps", "value": grad * 2})
            plan["reason"].append("batch already 1, increased grad accumulation as conservative OOM mitigation")

    if category == "schema" and cfg:
        dataset_name = str(cfg.get("dataset", ""))
        if dataset_name:
            info = _read_dataset_info(dataset_info_path)
            entry = info.get(dataset_name, {}) if isinstance(info.get(dataset_name, {}), dict) else {}
            data_dir = Path(str(cfg.get("dataset_dir", "data")))
            file_name = entry.get("file_name", "")
            first_row = _first_jsonl_record(data_dir / file_name) if file_name else None
            columns = entry.get("columns", {}) if isinstance(entry.get("columns", {}), dict) else {}
            ranking = bool(entry.get("ranking", False))
            desired: dict[str, str] = {}

            if isinstance(first_row, dict):
                if "prompt" in first_row and columns.get("prompt", "instruction") != "prompt":
                    desired["prompt"] = "prompt"
                if ranking:
                    if "chosen" in first_row and columns.get("chosen", "chosen") != "chosen":
                        desired["chosen"] = "chosen"
                    if "rejected" in first_row and columns.get("rejected", "rejected") != "rejected":
                        desired["rejected"] = "rejected"

            if desired:
                plan["actions"].append({
                    "type": "ensure_dataset_columns",
                    "dataset": dataset_name,
                    "columns": desired,
                })
                plan["reason"].append("fixed dataset columns mapping for observed row schema")

    if category == "lora" and cfg:
        model_name = str(cfg.get("model_name_or_path", "")).lower()
        if "gpt2" in model_name and str(cfg.get("lora_target", "")).lower() == "all":
            plan["actions"].append({"type": "set_yaml_key", "key": "lora_target", "value": "c_attn,c_proj,c_fc"})
            plan["reason"].append("set explicit GPT-2 friendly LoRA targets")

    return plan


def _llm_plan(log_text: str, api_base: str, model: str) -> dict[str, Any]:
    api_base = api_base.rstrip("/")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return JSON only with shape {'actions': [...]} using action types: "
                    "set_yaml_key(key,value), ensure_dataset_columns(dataset,columns). "
                    "Use only safe, low-risk fixes."
                ),
            },
            {
                "role": "user",
                "content": "Generate a safe auto-fix action plan from this log:\n\n" + log_text,
            },
        ],
        "temperature": 0.0,
    }

    resp = requests.post(f"{api_base}/chat/completions", json=payload, timeout=20)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]
    parsed = json.loads(raw)  # xray: ignore[PY-005]
    if not isinstance(parsed, dict) or not isinstance(parsed.get("actions"), list):
        raise ValueError("LLM did not return expected action-plan JSON object.")
    return parsed


def _merge_plans(primary: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(primary)
    seen = {json.dumps(a, sort_keys=True) for a in merged.get("actions", [])}
    for action in candidate.get("actions", []):
        if not isinstance(action, dict):
            continue
        if not _validate_action(action):
            continue
        key = json.dumps(action, sort_keys=True)
        if key not in seen:
            merged.setdefault("actions", []).append(action)
            seen.add(key)
    return merged


def _apply_actions(plan: dict[str, Any], config_path: Path | None, dataset_info_path: Path) -> list[str]:
    applied: list[str] = []

    cfg: dict[str, Any] | None = None
    if config_path and config_path.exists():
        cfg = _read_yaml(config_path)

    info = _read_dataset_info(dataset_info_path)

    for action in plan.get("actions", []):
        if not isinstance(action, dict) or not _validate_action(action):
            continue

        action_type = action["type"]
        if action_type == "set_yaml_key" and cfg is not None and config_path is not None:
            key = action["key"]
            value = action["value"]
            if cfg.get(key) != value:
                cfg[key] = value
                applied.append(f"set {config_path}:{key}={value}")

        elif action_type == "ensure_dataset_columns":
            dataset_name = action.get("dataset", "")
            columns = action.get("columns", {})
            if not dataset_name or dataset_name not in info or not isinstance(columns, dict):
                continue
            entry = info[dataset_name]
            if not isinstance(entry, dict):
                continue
            existing = entry.get("columns", {}) if isinstance(entry.get("columns", {}), dict) else {}
            changed = False
            for key, value in columns.items():
                if key in _ALLOWED_COLUMNS and isinstance(value, str) and existing.get(key) != value:
                    existing[key] = value
                    changed = True
            if changed:
                entry["columns"] = existing
                info[dataset_name] = entry
                applied.append(f"updated dataset_info columns for {dataset_name}")

    if cfg is not None and config_path is not None:
        _write_yaml(config_path, cfg)

    _write_dataset_info(dataset_info_path, info)
    return applied


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-healing fixer loop for cocktail gate failures.")
    parser.add_argument("--step", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--dataset-info", default="data/dataset_info.json")
    parser.add_argument("--api-base", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--plan-out", default="")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    log_path = Path(args.log)
    config_path = Path(args.config) if args.config else None
    dataset_info_path = Path(args.dataset_info)

    if not log_path.exists():
        raise SystemExit(f"Log file does not exist: {log_path}")

    log_text = _read_log(log_path)
    plan = _rule_based_plan(
        step=args.step,
        log_text=log_text,
        config_path=config_path,
        dataset_info_path=dataset_info_path,
    )

    if args.api_base and args.model:
        try:
            llm = _llm_plan(log_text, args.api_base, args.model)
            plan = _merge_plans(plan, llm)
            plan.setdefault("reason", []).append("merged safe subset of LLM-proposed actions")
        except Exception as exc:  # xray: ignore[QUAL-011]
            plan.setdefault("reason", []).append(f"llm_plan_error={exc}")

    if args.plan_out:
        Path(args.plan_out).write_text(json.dumps(plan, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(json.dumps(plan, indent=2, ensure_ascii=True))  # xray: ignore[PY-004]

    if args.apply:
        applied = _apply_actions(plan, config_path, dataset_info_path)
        print(json.dumps({"applied": applied}, indent=2, ensure_ascii=True))  # xray: ignore[PY-004]


if __name__ == "__main__":
    main()
