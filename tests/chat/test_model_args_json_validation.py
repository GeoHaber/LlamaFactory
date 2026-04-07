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

import pytest

from llamafactory.hparams.model_args import SGLangArguments, VllmArguments


def test_vllm_config_accepts_valid_json_object() -> None:
    args = VllmArguments(vllm_config='{"gpu_memory_utilization": 0.6}')
    assert isinstance(args.vllm_config, dict)
    assert args.vllm_config["gpu_memory_utilization"] == 0.6


def test_vllm_config_rejects_invalid_json_object() -> None:
    with pytest.raises(ValueError, match="vllm_config"):
        VllmArguments(vllm_config='{"gpu_memory_utilization": 0.6')


def test_sglang_config_accepts_valid_json_object() -> None:
    args = SGLangArguments(sglang_config='{"chunked_prefill_size": 1024}')
    assert isinstance(args.sglang_config, dict)
    assert args.sglang_config["chunked_prefill_size"] == 1024


def test_sglang_config_rejects_invalid_json_object() -> None:
    with pytest.raises(ValueError, match="sglang_config"):
        SGLangArguments(sglang_config='{"chunked_prefill_size": 1024')
