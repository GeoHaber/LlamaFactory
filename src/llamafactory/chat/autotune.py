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

from dataclasses import dataclass
import re
from typing import Literal, Optional

import torch

from ..extras.misc import get_current_memory, get_device_count, get_device_name, infer_optim_dtype
from ..extras.packages import is_kt_available, is_sglang_available, is_vllm_available

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


Policy = Literal["fast", "balanced", "quality"]


@dataclass
class RuntimeRecommendation:
    infer_backend: str
    infer_dtype: str
    ad_policy: Policy
    quantization_bit: Optional[int]
    max_concurrent: int
    summary: str


class HardwareAutoTuner:
    r"""Simple runtime auto tuner for backend/dtype/policy decisions.

    Heuristics are intentionally conservative to avoid OOM and to keep behavior
    predictable on mixed environments.
    """

    @staticmethod
    def _device_memory_gb() -> float:
        _, total = get_current_memory()
        if total <= 0:
            return 0.0

        return total / (1024**3)

    @staticmethod
    def _system_memory_gb() -> float:
        if psutil is None:
            return 0.0

        return psutil.virtual_memory().total / (1024**3)

    @staticmethod
    def _dtype_as_str() -> str:
        optimal = infer_optim_dtype(None)
        if optimal == torch.bfloat16:
            return "bfloat16"
        if optimal == torch.float16:
            return "float16"
        return "float32"

    @classmethod
    def recommend(cls, preferred_policy: Policy = "balanced") -> RuntimeRecommendation:
        device_name = get_device_name()
        device_count = get_device_count()
        device_mem_gb = cls._device_memory_gb()
        system_mem_gb = cls._system_memory_gb()

        infer_backend = "huggingface"
        infer_dtype = cls._dtype_as_str()
        ad_policy: Policy = preferred_policy
        quantization_bit: Optional[int] = None
        max_concurrent = 1

        if device_name == "gpu":
            if is_vllm_available() and device_mem_gb >= 16:
                infer_backend = "vllm"
            elif is_sglang_available() and device_mem_gb >= 12:
                infer_backend = "sglang"
            elif is_kt_available() and device_mem_gb <= 10:
                infer_backend = "ktransformers"

            if device_mem_gb < 8:
                quantization_bit = 4
                ad_policy = "fast"
                max_concurrent = 1
            elif device_mem_gb < 16:
                quantization_bit = 4
                ad_policy = "balanced" if preferred_policy != "quality" else "balanced"
                max_concurrent = 1
            elif device_mem_gb < 24:
                quantization_bit = None
                ad_policy = "balanced"
                max_concurrent = 2
            else:
                quantization_bit = None
                ad_policy = preferred_policy
                max_concurrent = 4 if device_count > 0 else 2

        elif device_name == "npu":
            infer_backend = "sglang" if is_sglang_available() else "huggingface"
            quantization_bit = 4 if device_mem_gb and device_mem_gb < 16 else None
            ad_policy = "balanced" if preferred_policy == "quality" else preferred_policy
            max_concurrent = 1 if device_mem_gb and device_mem_gb < 24 else 2

        elif device_name == "mps":
            infer_backend = "huggingface"
            infer_dtype = "float16"
            quantization_bit = 4 if system_mem_gb and system_mem_gb < 32 else None
            ad_policy = "fast" if system_mem_gb and system_mem_gb < 24 else "balanced"
            max_concurrent = 1

        else:
            infer_backend = "ktransformers" if is_kt_available() else "huggingface"
            infer_dtype = "float32"
            quantization_bit = 4 if infer_backend == "ktransformers" else None
            ad_policy = "fast"
            max_concurrent = 1

        summary = (
            f"device={device_name} count={device_count} device_mem_gb={device_mem_gb:.1f} "
            f"sys_mem_gb={system_mem_gb:.1f} backend={infer_backend} dtype={infer_dtype} "
            f"policy={ad_policy} quant={quantization_bit or 'none'} max_concurrent={max_concurrent}"
        )

        return RuntimeRecommendation(
            infer_backend=infer_backend,
            infer_dtype=infer_dtype,
            ad_policy=ad_policy,
            quantization_bit=quantization_bit,
            max_concurrent=max_concurrent,
            summary=summary,
        )

    @staticmethod
    def parse_model_size_b(model_name_or_path: str) -> float | None:
        if not model_name_or_path:
            return None

        # Prefer explicit activated parameter notation (e.g. 30B-A3B -> 3B active)
        active = re.search(r"A(\d+(?:\.\d+)?)B", model_name_or_path, flags=re.IGNORECASE)
        if active:
            return float(active.group(1))

        plain = re.search(r"(\d+(?:\.\d+)?)B", model_name_or_path, flags=re.IGNORECASE)
        if plain:
            return float(plain.group(1))

        return None

    @staticmethod
    def estimate_model_memory_gb(size_b: float, infer_dtype: str, quantization_bit: int | None) -> float:
        if quantization_bit == 4:
            bytes_per_param = 0.5
        elif quantization_bit == 8:
            bytes_per_param = 1.0
        elif infer_dtype in ("float16", "bfloat16"):
            bytes_per_param = 2.0
        else:
            bytes_per_param = 4.0

        raw_gb = size_b * 1e9 * bytes_per_param / (1024**3)
        # Runtime overhead for KV cache, buffers, fragmentation.
        return raw_gb * 1.2

    @classmethod
    def predict_cocktail(
        cls,
        base_model_name_or_path: str,
        infer_dtype: str,
        quantization_bit: int | None,
        policy: Policy,
        use_role_specific_models: bool,
        planner_model: str | None = None,
        coder_model: str | None = None,
        logician_model: str | None = None,
    ) -> dict[str, float | str]:
        base_size = cls.parse_model_size_b(base_model_name_or_path) or 7.0
        device_mem_gb = cls._device_memory_gb()

        role_sizes = {
            "planner": cls.parse_model_size_b(planner_model or "") if use_role_specific_models else None,
            "coder": cls.parse_model_size_b(coder_model or "") if use_role_specific_models else None,
            "logician": cls.parse_model_size_b(logician_model or "") if use_role_specific_models else None,
        }

        effective_role_sizes = {
            role: (size if size is not None else base_size) for role, size in role_sizes.items()
        }

        if policy == "fast":
            active_roles = ["coder"]
        elif policy == "balanced":
            active_roles = ["planner", "coder"]
        else:
            active_roles = ["planner", "coder", "logician"]

        total_mem = 0.0
        for role in active_roles:
            total_mem += cls.estimate_model_memory_gb(
                effective_role_sizes[role],
                infer_dtype=infer_dtype,
                quantization_bit=quantization_bit,
            )

        fit_ratio = (device_mem_gb / total_mem) if total_mem > 0 else 0.0
        if fit_ratio >= 1.3:
            speed_tier = "high"
            efficiency = 90
        elif fit_ratio >= 1.0:
            speed_tier = "medium"
            efficiency = 75
        elif fit_ratio >= 0.8:
            speed_tier = "constrained"
            efficiency = 55
        else:
            speed_tier = "overcommitted"
            efficiency = 35

        if speed_tier == "overcommitted":
            complexity_threshold = 900
            suggested_policy: Policy = "fast"
        elif speed_tier == "constrained" and policy == "quality":
            complexity_threshold = 700
            suggested_policy = "balanced"
        elif policy == "fast":
            complexity_threshold = 800
            suggested_policy = "fast"
        elif policy == "balanced":
            complexity_threshold = 450
            suggested_policy = "balanced"
        else:
            complexity_threshold = 250
            suggested_policy = "quality"

        return {
            "base_model_size_b": round(base_size, 2),
            "estimated_total_memory_gb": round(total_mem, 2),
            "available_device_memory_gb": round(device_mem_gb, 2),
            "fit_ratio": round(fit_ratio, 2),
            "speed_tier": speed_tier,
            "efficiency_score": efficiency,
            "suggested_policy": suggested_policy,
            "suggested_complexity_threshold": complexity_threshold,
        }
