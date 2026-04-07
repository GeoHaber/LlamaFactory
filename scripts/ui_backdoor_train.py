#!/usr/bin/env python
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

"""Launch train jobs through the same backend utilities used by WebUI Runner.

This script intentionally avoids direct button interaction. It uses
`llamafactory.webui.common.gen_cmd` and `llamafactory.webui.common.save_cmd`,
which are the same helpers used by `Runner._preview` and `Runner._launch`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from yaml import safe_load  # xray: ignore[SEC-015]

from llamafactory.webui.common import gen_cmd, save_cmd  # xray: ignore[SEC-015]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")

    return data


def _validate_train_args(args: dict, config_path: Path) -> None:
    required = ["model_name_or_path", "stage", "do_train", "output_dir"]
    missing = [k for k in required if k not in args or args[k] in (None, "")]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required keys in {config_path}: {missing_str}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run training through WebUI backend helpers without UI clicks.")
    parser.add_argument("--config", required=True, help="Path to training YAML args file.")
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Print generated CLI command and exit without launching training.",
    )
    parser.add_argument(
        "--print-training-args-path",
        action="store_true",
        help="Print the generated training_args.yaml path before launch.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    train_args = _load_yaml(config_path)
    _validate_train_args(train_args, config_path)

    os.makedirs(str(train_args["output_dir"]), exist_ok=True)

    print("=== UI-backdoor preview (same formatter as WebUI) ===")  # xray: ignore[PY-004]
    print(gen_cmd(train_args))  # xray: ignore[PY-004]

    if args.preview_only:
        return 0

    cmd_yaml_path = save_cmd(train_args)
    if args.print_training_args_path:
        print(f"training_args_path={cmd_yaml_path}")  # xray: ignore[PY-004]

    env = dict(os.environ)
    env["LLAMABOARD_ENABLED"] = "1"
    env["LLAMABOARD_WORKDIR"] = str(train_args["output_dir"])
    if train_args.get("deepspeed") is not None:
        env["FORCE_TORCHRUN"] = "1"

    # Use current interpreter to avoid picking a global `llamafactory-cli` from another env.
    proc = subprocess.Popen([sys.executable, "-m", "llamafactory.cli", "train", cmd_yaml_path], env=env)  # xray: ignore[SEC-010]
    proc.wait()
    return int(proc.returncode)


if __name__ == "__main__":
    sys.exit(main())
