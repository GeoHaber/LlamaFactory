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

from unittest.mock import patch

from llamafactory.chat.autotune import HardwareAutoTuner


def test_autotune_cpu_prefers_safe_defaults():
    with patch("llamafactory.chat.autotune.get_device_name", return_value="cpu"), patch(
        "llamafactory.chat.autotune.get_device_count", return_value=0
    ), patch("llamafactory.chat.autotune.is_kt_available", return_value=False):
        rec = HardwareAutoTuner.recommend("quality")

    assert rec.infer_backend == "huggingface"
    assert rec.infer_dtype == "float32"
    assert rec.ad_policy == "fast"
    assert rec.quantization_bit is None


def test_autotune_low_vram_gpu_prefers_quantized_fast_mode():
    with patch("llamafactory.chat.autotune.get_device_name", return_value="gpu"), patch(
        "llamafactory.chat.autotune.get_device_count", return_value=1
    ), patch("llamafactory.chat.autotune.HardwareAutoTuner._device_memory_gb", return_value=7.5), patch(
        "llamafactory.chat.autotune.is_vllm_available", return_value=False
    ), patch("llamafactory.chat.autotune.is_sglang_available", return_value=False), patch(
        "llamafactory.chat.autotune.is_kt_available", return_value=True
    ):
        rec = HardwareAutoTuner.recommend("quality")

    assert rec.quantization_bit == 4
    assert rec.ad_policy == "fast"
    assert rec.max_concurrent == 1


def test_autotune_high_vram_gpu_prefers_quality_backend():
    with patch("llamafactory.chat.autotune.get_device_name", return_value="gpu"), patch(
        "llamafactory.chat.autotune.get_device_count", return_value=1
    ), patch("llamafactory.chat.autotune.HardwareAutoTuner._device_memory_gb", return_value=32.0), patch(
        "llamafactory.chat.autotune.is_vllm_available", return_value=True
    ):
        rec = HardwareAutoTuner.recommend("quality")

    assert rec.infer_backend == "vllm"
    assert rec.ad_policy == "quality"
    assert rec.quantization_bit is None
