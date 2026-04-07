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

import os
import sys
from pathlib import Path

import pytest

from llamafactory.extras.ploting import plot_loss
from llamafactory.hparams.training_args import RayArguments
from llamafactory.v1.config.arg_utils import get_plugin_config


def test_ray_init_kwargs_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="ray_init_kwargs"):
        RayArguments(ray_init_kwargs='{"num_cpus": 4')


def test_plugin_config_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="valid JSON"):
        get_plugin_config('{"name": "foo"')


def test_plot_loss_rejects_malformed_trainer_state_json(tmp_path: Path) -> None:
    trainer_state = tmp_path / "trainer_state.json"
    trainer_state.write_text('{"log_history": [', encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed trainer state JSON"):
        plot_loss(str(tmp_path), ["loss"])


def test_hparams_parser_rejects_malformed_json_file(tmp_path: Path) -> None:
    os.environ["DISABLE_VERSION_CHECK"] = "1"
    json_file = tmp_path / "bad_args.json"
    json_file.write_text('{"model_name_or_path": "Qwen"', encoding="utf-8")

    from llamafactory.hparams import parser as hparams_parser

    original_argv = sys.argv
    try:
        sys.argv = ["prog", str(json_file)]
        with pytest.raises(ValueError, match="Malformed JSON config file"):
            hparams_parser.read_args()
    finally:
        sys.argv = original_argv


def test_v1_arg_parser_rejects_malformed_json_file(tmp_path: Path) -> None:
    json_file = tmp_path / "bad_v1_args.json"
    json_file.write_text('{"seed": 42', encoding="utf-8")

    from llamafactory.v1.config import arg_parser as v1_arg_parser

    original_argv = sys.argv
    try:
        sys.argv = ["prog", str(json_file)]
        with pytest.raises(ValueError, match="Malformed JSON config file"):
            v1_arg_parser.get_args(None)
    finally:
        sys.argv = original_argv
