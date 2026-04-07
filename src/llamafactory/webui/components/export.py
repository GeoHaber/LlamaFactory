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

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

from ...extras.constants import PEFT_METHODS
from ...extras.misc import torch_gc
from ...extras.packages import is_gradio_available
from ...train.tuner import export_model
from ..common import get_save_dir, load_config
from ..locales import ALERTS


if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component

    from ..engine import Engine


GPTQ_BITS = ["8", "4", "3", "2"]
GGUF_BITS = ["F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"]
GGUF_PRESET_TO_BIT = {"quality": "Q8_0", "balanced": "Q5_K_M", "fastest": "Q4_K_M"}


def _detect_template_for_model(model_path: str, current_template: str) -> str:
    model_lc = (model_path or "").lower()
    if "qwen3" in model_lc:
        return "qwen3_nothink"
    if "qwen" in model_lc:
        return "qwen"
    if "llama3" in model_lc or "llama-3" in model_lc:
        return "llama3"
    if "llama" in model_lc:
        return "llama2"
    if "mistral" in model_lc:
        return "mistral"
    if "gemma" in model_lc:
        return "gemma"
    return current_template


def _update_detected_template(model_path: str, current_template: str) -> "gr.Textbox":
    return gr.Textbox(value=_detect_template_for_model(model_path, current_template), interactive=False)


def _resolve_gguf_bit(preset: str, use_raw: bool, raw_bit: str) -> str:
    if use_raw and raw_bit in GGUF_BITS:
        return raw_bit

    return GGUF_PRESET_TO_BIT.get(preset, GGUF_PRESET_TO_BIT["balanced"])


def _make_quantization_legend(lang: str) -> str:
    if lang == "ru":
        return (
            "### Обозначения GGUF\n"
            "<span style='color:#0ea5e9'><b>Q4/Q5/Q6/Q8</b></span>: число означает битность; больше бит обычно повышает качество, но требует больше памяти.\n\n"
            "<span style='color:#10b981'><b>_K</b></span>: формат K-quant с группировкой, часто стабильнее старых вариантов.\n\n"
            "<span style='color:#f59e0b'><b>_M</b></span>: смешанный вариант для практичного баланса качества и скорости.\n\n"
            "<span style='color:#a78bfa'><b>F16</b></span>: почти полная точность, максимальное качество и самый большой размер."
        )

    if lang == "ko":
        return (
            "### GGUF 기호 안내\n"
            "<span style='color:#0ea5e9'><b>Q4/Q5/Q6/Q8</b></span>: 숫자는 비트 폭을 뜻하며, 비트가 높을수록 품질은 좋아지지만 메모리 사용이 늘어납니다.\n\n"
            "<span style='color:#10b981'><b>_K</b></span>: K-quant 그룹 형식으로, 구형 양자화보다 안정적인 경우가 많습니다.\n\n"
            "<span style='color:#f59e0b'><b>_M</b></span>: 실사용에서 품질/속도 균형을 맞춘 혼합 변형입니다.\n\n"
            "<span style='color:#a78bfa'><b>F16</b></span>: 거의 완전 정밀도이며, 품질은 가장 높고 크기도 가장 큽니다."
        )

    if lang == "ja":
        return (
            "### GGUF 記号ガイド\n"
            "<span style='color:#0ea5e9'><b>Q4/Q5/Q6/Q8</b></span>: 数字はビット幅を示し、通常はビット数が高いほど品質が上がる一方でメモリ使用量も増えます。\n\n"
            "<span style='color:#10b981'><b>_K</b></span>: K-quant のグループ化形式で、従来形式より安定しやすい傾向があります。\n\n"
            "<span style='color:#f59e0b'><b>_M</b></span>: 実運用での品質と速度のバランスを狙った混合バリアントです。\n\n"
            "<span style='color:#a78bfa'><b>F16</b></span>: ほぼフル精度で、品質は最も高くサイズも最大です。"
        )

    if lang == "zh":
        return (
            "### GGUF 符号说明\n"
            "<span style='color:#0ea5e9'><b>Q4/Q5/Q6/Q8</b></span>：数字表示位宽，位数越高通常质量越高、速度越慢。\n\n"
            "<span style='color:#10b981'><b>_K</b></span>：K-quant 分组方案，通常比旧式量化更稳。\n\n"
            "<span style='color:#f59e0b'><b>_M</b></span>：混合策略变体，常用于质量/速度平衡。\n\n"
            "<span style='color:#a78bfa'><b>F16</b></span>：非整数量化，质量最高但体积最大。"
        )

    return (
        "### GGUF Symbol Guide\n"
        "<span style='color:#0ea5e9'><b>Q4/Q5/Q6/Q8</b></span>: number indicates bit-width; higher bits usually improve quality but use more memory.\n\n"
        "<span style='color:#10b981'><b>_K</b></span>: K-quant grouped format, often more stable than older quant formats.\n\n"
        "<span style='color:#f59e0b'><b>_M</b></span>: mixed variant tuned for practical quality/speed balance.\n\n"
        "<span style='color:#a78bfa'><b>F16</b></span>: full/near-full precision, highest quality and largest size."
    )


def _make_quantization_help(lang: str, preset: str, quantization_bit: str) -> str:
    heading = {
        "en": "### GGUF Quantization Guide",
        "zh": "### GGUF 量化说明",
        "ru": "### Руководство по квантованию GGUF",
        "ko": "### GGUF 양자화 가이드",
        "ja": "### GGUF 量子化ガイド",
    }.get(lang, "### GGUF Quantization Guide")

    preset_desc = {
        "quality": {
            "en": "Quality: larger model, best fidelity.",
            "zh": "高质量：体积更大，精度更高。",
            "ru": "Качество: модель больше, наилучшая точность.",
            "ko": "고품질: 모델 크기는 크지만 정확도가 가장 좋습니다.",
            "ja": "高品質: モデルサイズは大きいが再現性が最も高い。",
        },
        "balanced": {
            "en": "Balanced: good speed/quality trade-off.",
            "zh": "平衡：速度与质量折中。",
            "ru": "Баланс: оптимальный компромисс между скоростью и качеством.",
            "ko": "균형: 속도와 품질의 균형이 좋습니다.",
            "ja": "バランス: 速度と品質のバランスが良い。",
        },
        "fastest": {
            "en": "Fastest: smaller model, fastest local inference.",
            "zh": "极速：体积更小，本地推理更快。",
            "ru": "Максимальная скорость: меньший размер и самая быстрая локальная инференция.",
            "ko": "최고 속도: 모델이 더 작고 로컬 추론이 가장 빠릅니다.",
            "ja": "最速: モデルが小さく、ローカル推論が最速。",
        },
    }
    desc = preset_desc.get(preset, preset_desc["balanced"]).get(lang, preset_desc.get(preset, preset_desc["balanced"])["en"])

    preset_name = {
        "quality": {"en": "quality", "zh": "高质量", "ru": "качество", "ko": "고품질", "ja": "高品質"},
        "balanced": {"en": "balanced", "zh": "平衡", "ru": "баланс", "ko": "균형", "ja": "バランス"},
        "fastest": {"en": "fastest", "zh": "极速", "ru": "макс. скорость", "ko": "최고 속도", "ja": "最速"},
        "custom": {"en": "custom", "zh": "自定义", "ru": "пользовательский", "ko": "사용자 지정", "ja": "カスタム"},
    }.get(preset, {"en": preset}).get(lang, preset)

    if lang == "zh":
        explain = "- `Q4/Q5/Q6/Q8` 中的数字表示权重位宽（位数越高质量越好、速度越慢）。\n- `_K` 表示 K-quant 分组方案。\n- `_M` 常见为中等混合策略。"
    elif lang == "ru":
        explain = "- Цифры в `Q4/Q5/Q6/Q8` обозначают битность (больше бит обычно повышает качество, но требует больше памяти).\n- `_K` означает схему группировки K-quant.\n- `_M` обычно означает смешанный вариант для баланса качества и скорости."
    elif lang == "ko":
        explain = "- `Q4/Q5/Q6/Q8`의 숫자는 비트 폭을 의미합니다 (비트가 높을수록 품질은 좋아지지만 메모리 사용이 늘어납니다).\n- `_K`는 K-quant 그룹화 방식을 뜻합니다.\n- `_M`은 품질/속도 균형을 위한 혼합 변형입니다."
    elif lang == "ja":
        explain = "- `Q4/Q5/Q6/Q8` の数字はビット幅を示します（通常、ビットが高いほど品質は向上しますがメモリ使用量も増えます）。\n- `_K` は K-quant のグループ化方式です。\n- `_M` は品質と速度のバランスを狙った混合バリアントです。"
    else:
        explain = "- Digits in `Q4/Q5/Q6/Q8` indicate bit-width (higher bits usually improve quality but use more memory).\n- `_K` means K-quant grouping scheme.\n- `_M` is a mixed variant with practical quality/speed balance."

    preset_label = {"zh": "预设", "ru": "Профиль", "ko": "프리셋", "ja": "プリセット"}.get(lang, "Preset")
    quant_label = {"zh": "量化", "ru": "Квантование", "ko": "양자화", "ja": "量子化"}.get(lang, "Quantization")
    return f"{heading}\n\n- {preset_label}: **{preset_name}**\n- {quant_label}: **{quantization_bit}**\n- {desc}\n\n{explain}"


def _make_export_help(lang: str) -> str:
    if lang == "zh":
        return (
            "### 导出菜单帮助（简单版）\n\n"
            "- **Max shard size (GB)**\n"
            "  是什么：每个导出文件的最大大小。\n"
            "  为什么：文件更小更容易上传和复制。\n"
            "  怎么用：一般保持默认，除非你的存储或平台有单文件大小限制。\n\n"
            "- **Export quantization bit**\n"
            "  是什么：普通导出时的量化位数。\n"
            "  为什么：位数越低体积越小、内存更省，但质量可能下降。\n"
            "  怎么用：追求质量用 `none`，追求体积再降位。\n\n"
            "- **Export quantization dataset**\n"
            "  是什么：量化校准数据集。\n"
            "  为什么：帮助量化后尽量保持模型行为。\n"
            "  怎么用：仅在启用量化时填写。\n\n"
            "- **Export device**\n"
            "  是什么：导出时使用的设备。\n"
            "  为什么：影响速度和内存压力。\n"
            "  怎么用：稳定优先选 `cpu`，自动选择用 `auto`。\n\n"
            "- **Export legacy format**\n"
            "  是什么：用旧格式保存（非 safetensors）。\n"
            "  为什么：兼容旧工具。\n"
            "  怎么用：除非下游工具明确要求，否则保持关闭。\n\n"
            "- **Export dir**\n"
            "  是什么：导出文件保存目录。\n"
            "  为什么：决定产物写到哪里。\n"
            "  怎么用：确保目标盘空间充足。\n\n"
            "- **HF Hub ID (optional)**\n"
            "  是什么：Hugging Face 仓库名。\n"
            "  为什么：可直接上传到 Hub。\n"
            "  怎么用：仅本地导出就留空。\n\n"
            "- **Extra arguments (JSON)**\n"
            "  是什么：传给导出器的高级参数。\n"
            "  为什么：做精细控制。\n"
            "  怎么用：不确定就保持 `{}`。"
        )

    return (
        "### Export Menu Help (simple)\n\n"
        "- **Max shard size (GB)**\n"
        "  What: max file size per exported chunk.\n"
        "  Why: smaller files are easier to upload and move.\n"
        "  How: keep default unless your storage target has per-file limits.\n\n"
        "- **Export quantization bit**\n"
        "  What: quantization level for normal export.\n"
        "  Why: lower bits shrink size and RAM, but can reduce quality.\n"
        "  How: use `none` for quality-first, lower bits for size-first.\n\n"
        "- **Export quantization dataset**\n"
        "  What: calibration dataset path used during quantization.\n"
        "  Why: helps preserve behavior after quantization.\n"
        "  How: needed only when quantization is enabled.\n\n"
        "- **Export device**\n"
        "  What: hardware used during export.\n"
        "  Why: impacts speed and memory pressure.\n"
        "  How: choose `cpu` for reliability, `auto` to let the system decide.\n\n"
        "- **Export legacy format**\n"
        "  What: save in old format instead of safetensors.\n"
        "  Why: compatibility with older downstream tools.\n"
        "  How: leave off unless a tool explicitly requires it.\n\n"
        "- **Export dir**\n"
        "  What: output folder for exported files.\n"
        "  Why: controls where artifacts are written.\n"
        "  How: pick a folder with enough free disk space.\n\n"
        "- **HF Hub ID (optional)**\n"
        "  What: Hugging Face repository ID.\n"
        "  Why: enables direct upload to Hub.\n"
        "  How: leave empty for local-only export.\n\n"
        "- **Extra arguments (JSON)**\n"
        "  What: advanced export arguments.\n"
        "  Why: gives fine-grained control.\n"
        "  How: keep `{}` unless you know exactly what to override."
    )


def _make_gguf_help(lang: str) -> str:
    if lang == "zh":
        return (
            "### GGUF 菜单帮助（简单版）\n\n"
            "- **GGUF merge export dir**：转换前的合并模型目录。\n"
            "  为什么：GGUF 转换需要完整合并模型。\n"
            "  怎么用：首次导出保留默认。\n\n"
            "- **Auto-detect template / Detected template**：自动识别并显示模板。\n"
            "  为什么：模板不对会导致对话格式异常。\n"
            "  怎么用：默认开启，除非你确认要手动覆盖。\n\n"
            "- **llama.cpp dir**：llama.cpp 路径。\n"
            "  为什么：这里有转换和量化工具。\n"
            "  怎么用：指向包含 `convert_hf_to_gguf.py` 和 `llama-quantize` 的目录。\n\n"
            "- **GGUF output file**：最终 `.gguf` 路径。\n"
            "  为什么：这是你后续加载推理的文件。\n"
            "  怎么用：可直接使用建议文件名。\n\n"
            "- **GGUF preset**：质量预设。\n"
            "  为什么：比手选量化类型更安全。\n"
            "  怎么用：先用 `balanced`，质量优先用 `quality`，速度优先用 `fastest`。\n\n"
            "- **Use raw quantization override / Raw GGUF type**：高级手动量化。\n"
            "  为什么：需要精确控制时使用。\n"
            "  怎么用：只有你明确知道目标量化类型时再开启。\n\n"
            "- **Skip merge/export (GGUF-only)**：跳过合并，仅转换/量化。\n"
            "  为什么：已有合并模型时更快。\n"
            "  怎么用：仅在合并目录已经准备好时开启。"
        )

    return (
        "### GGUF Menu Help (simple)\n\n"
        "- **GGUF merge export dir**: temporary merged-model folder before conversion.\n"
        "  Why: converter needs a merged full model.\n"
        "  How: keep default for first run.\n\n"
        "- **Auto-detect template / Detected template**: auto-select and show template.\n"
        "  Why: wrong template often breaks chat prompt format.\n"
        "  How: keep auto-detect on unless you know a manual override is required.\n\n"
        "- **llama.cpp dir**: path to llama.cpp tools.\n"
        "  Why: conversion and quantization executables are loaded from here.\n"
        "  How: point to folder with `convert_hf_to_gguf.py` and `llama-quantize`.\n\n"
        "- **GGUF output file**: final `.gguf` path.\n"
        "  Why: this is the model artifact you run later.\n"
        "  How: use suggested filename unless you need custom naming.\n\n"
        "- **GGUF preset**: quality profile.\n"
        "  Why: easier and safer than low-level quant names.\n"
        "  How: start with `balanced`, use `quality` for better output, `fastest` for lower RAM.\n\n"
        "- **Use raw quantization override / Raw GGUF type**: manual advanced control.\n"
        "  Why: useful for exact reproducibility or expert tuning.\n"
        "  How: enable only if you already know your target quantization type.\n\n"
        "- **Skip merge/export (GGUF-only)**: skip merge step and only convert/quantize.\n"
        "  Why: saves time when merged export already exists.\n"
        "  How: enable only when merge export dir already has a valid merged model."
    )


def _update_help_text(lang: str) -> tuple["gr.Markdown", "gr.Markdown"]:
    return gr.Markdown(value=_make_export_help(lang)), gr.Markdown(value=_make_gguf_help(lang))


def _update_gguf_recommendations(
    lang: str, preset: str, use_raw_quantization: bool, raw_quantization: str, gguf_export_dir: str
) -> tuple["gr.Textbox", "gr.Markdown", "gr.Markdown"]:
    quantization_bit = _resolve_gguf_bit(preset, use_raw_quantization, raw_quantization)
    effective_preset = "custom" if use_raw_quantization else preset
    out_dir = gguf_export_dir.strip() if gguf_export_dir else "saves/gguf_export"
    suffix = quantization_bit.lower()
    suggested = Path(out_dir).as_posix().rstrip("/") + f"/model_{suffix}.gguf"
    return (
        gr.Textbox(value=suggested),
        gr.Markdown(value=_make_quantization_help(lang, effective_preset, quantization_bit)),
        gr.Markdown(value=_make_quantization_legend(lang)),
    )


def _resolve_adapter_path(model_name: str, finetuning_type: str, checkpoint_path: str | list[str]) -> str | None:
    if not checkpoint_path:
        return None

    if finetuning_type in PEFT_METHODS and isinstance(checkpoint_path, list):
        return ",".join([get_save_dir(model_name, finetuning_type, adapter) for adapter in checkpoint_path])

    if isinstance(checkpoint_path, str):
        return get_save_dir(model_name, finetuning_type, checkpoint_path)

    return None


def _get_pwsh_executable() -> str | None:
    return shutil.which("pwsh") or shutil.which("powershell")


def _list_llama_cpp_dirs() -> list[str]:
    candidates = [
        Path("llama.cpp"),
        Path("..") / "llama.cpp",
        Path("D:/src/llama.cpp"),
        Path("C:/src/llama.cpp"),
    ]
    choices: list[str] = []
    for path in candidates:
        if path.exists():
            choices.append(path.resolve().as_posix())

    return choices


def _suggest_gguf_outfile(gguf_export_dir: str, gguf_quantization: str) -> "gr.Textbox":
    out_dir = gguf_export_dir.strip() if gguf_export_dir else "saves/gguf_export"
    suffix = gguf_quantization.lower()
    suggested = Path(out_dir).as_posix().rstrip("/") + f"/model_{suffix}.gguf"
    return gr.Textbox(value=suggested)


def can_quantize(checkpoint_path: str | list[str]) -> "gr.Dropdown":
    if isinstance(checkpoint_path, list) and len(checkpoint_path) != 0:
        return gr.Dropdown(value="none", interactive=False)
    else:
        return gr.Dropdown(interactive=True)


def save_model(
    lang: str,
    model_name: str,
    model_path: str,
    finetuning_type: str,
    checkpoint_path: str | list[str],
    template: str,
    export_size: int,
    export_quantization_bit: str,
    export_quantization_dataset: str,
    export_device: str,
    export_legacy_format: bool,
    export_dir: str,
    export_hub_model_id: str,
    extra_args: str,
) -> Generator[str, None, None]:
    user_config = load_config()
    error = ""
    if not model_name:
        error = ALERTS["err_no_model"][lang]
    elif not model_path:
        error = ALERTS["err_no_path"][lang]
    elif not export_dir:
        error = ALERTS["err_no_export_dir"][lang]
    elif export_quantization_bit in GPTQ_BITS and not export_quantization_dataset:
        error = ALERTS["err_no_dataset"][lang]
    elif export_quantization_bit not in GPTQ_BITS and not checkpoint_path:
        error = ALERTS["err_no_adapter"][lang]
    elif export_quantization_bit in GPTQ_BITS and checkpoint_path and isinstance(checkpoint_path, list):
        error = ALERTS["err_gptq_lora"][lang]

    try:
        extra_args_dict = json.loads(extra_args)
    except json.JSONDecodeError:
        error = ALERTS["err_json_schema"][lang]

    if error:
        gr.Warning(error)
        yield error
        return

    args = dict(
        model_name_or_path=model_path,
        cache_dir=user_config.get("cache_dir", None),
        finetuning_type=finetuning_type,
        template=template,
        export_dir=export_dir,
        export_hub_model_id=export_hub_model_id or None,
        export_size=export_size,
        export_quantization_bit=int(export_quantization_bit) if export_quantization_bit in GPTQ_BITS else None,
        export_quantization_dataset=export_quantization_dataset,
        export_device=export_device,
        export_legacy_format=export_legacy_format,
        trust_remote_code=True,
    )
    args.update(extra_args_dict)

    adapter_path = _resolve_adapter_path(model_name, finetuning_type, checkpoint_path)
    if adapter_path:
        if finetuning_type in PEFT_METHODS and isinstance(checkpoint_path, list):
            args["adapter_name_or_path"] = adapter_path
        else:
            args["model_name_or_path"] = adapter_path

    yield ALERTS["info_exporting"][lang]
    export_model(args)
    torch_gc()
    yield ALERTS["info_exported"][lang]


def export_gguf(
    lang: str,
    model_name: str,
    model_path: str,
    finetuning_type: str,
    checkpoint_path: str | list[str],
    template: str,
    gguf_use_detected_template: bool,
    gguf_detected_template: str,
    gguf_preset: str,
    gguf_use_raw_quantization: bool,
    gguf_raw_quantization: str,
    gguf_export_dir: str,
    gguf_llama_cpp_dir: str,
    gguf_outfile: str,
    gguf_skip_merge: bool,
) -> Generator[str, None, None]:
    error = ""
    gguf_quantization = _resolve_gguf_bit(gguf_preset, gguf_use_raw_quantization, gguf_raw_quantization)
    selected_template = gguf_detected_template if gguf_use_detected_template and gguf_detected_template else template
    if os.name != "nt":
        error = ALERTS["err_gguf_windows_only"][lang]
    elif not model_name:
        error = ALERTS["err_no_model"][lang]
    elif not model_path:
        error = ALERTS["err_no_path"][lang]
    elif not gguf_export_dir:
        error = ALERTS["err_no_export_dir"][lang]
    elif not gguf_skip_merge and finetuning_type in PEFT_METHODS and not checkpoint_path:
        error = ALERTS["err_no_adapter"][lang]
    elif not gguf_llama_cpp_dir:
        error = ALERTS["err_no_llama_cpp_dir"][lang]
    elif not Path(gguf_llama_cpp_dir).exists():
        error = ALERTS["err_no_llama_cpp_dir"][lang]
    elif not gguf_outfile:
        error = ALERTS["err_no_gguf_outfile"][lang]
    elif not selected_template:
        error = ALERTS["err_no_template"][lang]

    if error:
        gr.Warning(error)
        yield error
        return

    pwsh = _get_pwsh_executable()
    if not pwsh:
        msg = ALERTS["err_no_powershell"][lang]
        gr.Warning(msg)
        yield msg
        return

    script_path = Path(__file__).resolve().parents[4] / "scripts" / "export_local_gguf.ps1"
    if not script_path.exists():
        msg = ALERTS["err_no_gguf_script"][lang]
        gr.Warning(msg)
        yield msg
        return

    cmd = [
        pwsh,
        "-File",
        str(script_path),
        "-PythonExe",
        sys.executable,
        "-BaseModel",
        model_path,
        "-Template",
        selected_template,
        "-ExportDir",
        gguf_export_dir,
        "-LlamaCppDir",
        gguf_llama_cpp_dir,
        "-OutFile",
        gguf_outfile,
        "-Quantization",
        gguf_quantization,
    ]

    adapter_path = _resolve_adapter_path(model_name, finetuning_type, checkpoint_path)
    if adapter_path and finetuning_type in PEFT_METHODS:
        cmd.extend(["-AdapterPath", adapter_path])

    if gguf_skip_merge:
        cmd.append("-SkipMergeExport")

    yield ALERTS["info_gguf_exporting"][lang] + f"\nTemplate: {selected_template}\nQuantization: {gguf_quantization}"
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    running_log = ""
    if proc.stdout is not None:
        for line in proc.stdout:
            running_log += line
            yield running_log

    return_code = proc.wait()
    if return_code != 0:
        error_msg = ALERTS["err_failed"][lang] + f" Exit code: {return_code}\n\n{running_log}"
        gr.Warning(error_msg)
        yield error_msg
        return

    yield ALERTS["info_gguf_exported"][lang]


def create_export_tab(engine: "Engine") -> dict[str, "Component"]:
    with gr.Row():
        export_size = gr.Slider(minimum=1, maximum=100, value=5, step=1)
        export_quantization_bit = gr.Dropdown(choices=["none"] + GPTQ_BITS, value="none")
        export_quantization_dataset = gr.Textbox(value="data/c4_demo.jsonl")
        export_device = gr.Radio(choices=["cpu", "auto"], value="cpu")
        export_legacy_format = gr.Checkbox()

    with gr.Row():
        export_dir = gr.Textbox()
        export_hub_model_id = gr.Textbox()
        extra_args = gr.Textbox(value="{}")

    with gr.Accordion("Export help", open=False, visible=False) as export_help_tab:
        export_help = gr.Markdown(value=_make_export_help("en"))

    with gr.Accordion("GGUF export", open=False) as gguf_tab:
        with gr.Row():
            gguf_use_detected_template = gr.Checkbox(value=True)
            gguf_detected_template = gr.Textbox(interactive=False)
        with gr.Row():
            gguf_export_dir = gr.Textbox(value="saves/gguf_export/merged_hf")
            gguf_llama_cpp_dir = gr.Dropdown(choices=_list_llama_cpp_dirs(), allow_custom_value=True)
        with gr.Row():
            gguf_outfile = gr.Textbox(value="saves/gguf_export/model_q5_k_m.gguf")
            gguf_preset = gr.Dropdown(choices=["quality", "balanced", "fastest"], value="balanced")
            gguf_skip_merge = gr.Checkbox(value=False)

        gguf_quantization_help = gr.Markdown(value=_make_quantization_help("en", "balanced", GGUF_PRESET_TO_BIT["balanced"]))
        with gr.Accordion("Advanced quantization", open=False) as gguf_advanced_tab:
            with gr.Row():
                gguf_use_raw_quantization = gr.Checkbox(value=False)
                gguf_raw_quantization = gr.Dropdown(choices=GGUF_BITS, value=GGUF_PRESET_TO_BIT["balanced"])

            gguf_quantization_legend = gr.Markdown(value=_make_quantization_legend("en"))

        gguf_btn = gr.Button()
        gguf_info_box = gr.Textbox(show_label=False, interactive=False)

    with gr.Accordion("GGUF help", open=False, visible=False) as gguf_help_tab:
        gguf_help = gr.Markdown(value=_make_gguf_help("en"))

    top_model_path = engine.manager.get_elem_by_id("top.model_path")
    top_template = engine.manager.get_elem_by_id("top.template")
    top_lang = engine.manager.get_elem_by_id("top.lang")

    top_model_path.change(_update_detected_template, [top_model_path, top_template], [gguf_detected_template], queue=False)
    top_template.change(_update_detected_template, [top_model_path, top_template], [gguf_detected_template], queue=False)

    gguf_export_dir.change(
        _update_gguf_recommendations,
        [top_lang, gguf_preset, gguf_use_raw_quantization, gguf_raw_quantization, gguf_export_dir],
        [gguf_outfile, gguf_quantization_help, gguf_quantization_legend],
        queue=False,
    )
    gguf_preset.change(
        _update_gguf_recommendations,
        [top_lang, gguf_preset, gguf_use_raw_quantization, gguf_raw_quantization, gguf_export_dir],
        [gguf_outfile, gguf_quantization_help, gguf_quantization_legend],
        queue=False,
    )
    gguf_use_raw_quantization.change(
        _update_gguf_recommendations,
        [top_lang, gguf_preset, gguf_use_raw_quantization, gguf_raw_quantization, gguf_export_dir],
        [gguf_outfile, gguf_quantization_help, gguf_quantization_legend],
        queue=False,
    )
    gguf_raw_quantization.change(
        _update_gguf_recommendations,
        [top_lang, gguf_preset, gguf_use_raw_quantization, gguf_raw_quantization, gguf_export_dir],
        [gguf_outfile, gguf_quantization_help, gguf_quantization_legend],
        queue=False,
    )
    top_lang.change(
        _update_gguf_recommendations,
        [top_lang, gguf_preset, gguf_use_raw_quantization, gguf_raw_quantization, gguf_export_dir],
        [gguf_outfile, gguf_quantization_help, gguf_quantization_legend],
        queue=False,
    )
    top_lang.change(_update_help_text, [top_lang], [export_help, gguf_help], queue=False)

    checkpoint_path: gr.Dropdown = engine.manager.get_elem_by_id("top.checkpoint_path")
    checkpoint_path.change(can_quantize, [checkpoint_path], [export_quantization_bit], queue=False)

    export_btn = gr.Button()
    info_box = gr.Textbox(show_label=False, interactive=False)

    export_btn.click(
        save_model,
        [
            engine.manager.get_elem_by_id("top.lang"),
            engine.manager.get_elem_by_id("top.model_name"),
            engine.manager.get_elem_by_id("top.model_path"),
            engine.manager.get_elem_by_id("top.finetuning_type"),
            engine.manager.get_elem_by_id("top.checkpoint_path"),
            engine.manager.get_elem_by_id("top.template"),
            export_size,
            export_quantization_bit,
            export_quantization_dataset,
            export_device,
            export_legacy_format,
            export_dir,
            export_hub_model_id,
            extra_args,
        ],
        [info_box],
    )

    gguf_btn.click(
        export_gguf,
        [
            engine.manager.get_elem_by_id("top.lang"),
            engine.manager.get_elem_by_id("top.model_name"),
            engine.manager.get_elem_by_id("top.model_path"),
            engine.manager.get_elem_by_id("top.finetuning_type"),
            engine.manager.get_elem_by_id("top.checkpoint_path"),
            engine.manager.get_elem_by_id("top.template"),
            gguf_use_detected_template,
            gguf_detected_template,
            gguf_preset,
            gguf_use_raw_quantization,
            gguf_raw_quantization,
            gguf_export_dir,
            gguf_llama_cpp_dir,
            gguf_outfile,
            gguf_skip_merge,
        ],
        [gguf_info_box],
    )

    return dict(
        export_size=export_size,
        export_quantization_bit=export_quantization_bit,
        export_quantization_dataset=export_quantization_dataset,
        export_device=export_device,
        export_legacy_format=export_legacy_format,
        export_dir=export_dir,
        export_hub_model_id=export_hub_model_id,
        extra_args=extra_args,
        export_help_tab=export_help_tab,
        export_help=export_help,
        gguf_export_dir=gguf_export_dir,
        gguf_tab=gguf_tab,
        gguf_help_tab=gguf_help_tab,
        gguf_help=gguf_help,
        gguf_llama_cpp_dir=gguf_llama_cpp_dir,
        gguf_use_detected_template=gguf_use_detected_template,
        gguf_detected_template=gguf_detected_template,
        gguf_preset=gguf_preset,
        gguf_use_raw_quantization=gguf_use_raw_quantization,
        gguf_raw_quantization=gguf_raw_quantization,
        gguf_outfile=gguf_outfile,
        gguf_quantization_help=gguf_quantization_help,
        gguf_quantization_legend=gguf_quantization_legend,
        gguf_advanced_tab=gguf_advanced_tab,
        gguf_skip_merge=gguf_skip_merge,
        gguf_btn=gguf_btn,
        gguf_info_box=gguf_info_box,
        export_btn=export_btn,
        info_box=info_box,
    )
