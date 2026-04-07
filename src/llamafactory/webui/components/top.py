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

from typing import TYPE_CHECKING

from ...data import TEMPLATES
from ...extras.constants import METHODS, SUPPORTED_MODELS
from ...extras.misc import use_modelscope, use_openmind
from ...extras.packages import is_gradio_available
from ..common import save_config
from ..control import can_quantize, can_quantize_to, check_template, get_model_info, list_checkpoints, switch_hub


if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component


SUPPORTED_WEBUI_LANGS = ["en", "ro", "hu", "he", "fr", "de", "es", "pt"]


def _make_top_help(lang: str) -> str:
    if lang == "zh":
        return (
            "### 顶部菜单帮助（简单版）\n\n"
            "- **Language**\n"
            "  What: 界面显示语言。\n"
            "  Why: 便于阅读和操作。\n"
            "  How: 直接切换即可。\n\n"
            "- **Model name / Model path**\n"
            "  What: 模型名称与路径。\n"
            "  Why: 决定后续训练/评估/导出使用哪个模型。\n"
            "  How: 已知模型可选名称，自定义模型填路径。\n\n"
            "- **Hub name**\n"
            "  What: 模型下载来源。\n"
            "  Why: 不同来源可用模型不同。\n"
            "  How: 按你账号和网络环境选择。\n\n"
            "- **Finetuning method / Checkpoint path**\n"
            "  What: 微调方法与断点目录。\n"
            "  Why: 影响是否加载 LoRA 适配器与继续训练。\n"
            "  How: 有历史断点就选对应路径。\n\n"
            "- **Quantization bit / method**\n"
            "  What: 量化位数与算法。\n"
            "  Why: 影响显存、速度与精度。\n"
            "  How: 默认最稳，显存不足再开启量化。\n\n"
            "- **Chat template / RoPE scaling / Booster**\n"
            "  What: 提示模板、上下文扩展、性能加速选项。\n"
            "  Why: 影响对话格式、长上下文和训练速度。\n"
            "  How: 不确定时保持默认。"
        )

    return (
        "### Top Controls Help (simple)\n\n"
        "- **Language**\n"
        "  What: UI display language.\n"
        "  Why: makes the interface easier to read.\n"
        "  How: choose your preferred language from the dropdown.\n\n"
        "- **Model name / Model path**\n"
        "  What: selected model identity and its location.\n"
        "  Why: all train/eval/export actions use this model.\n"
        "  How: pick a known model name, or set a custom path.\n\n"
        "- **Hub name**\n"
        "  What: model download source.\n"
        "  Why: different sources expose different model mirrors.\n"
        "  How: select the source that works best for your setup.\n\n"
        "- **Finetuning method / Checkpoint path**\n"
        "  What: tuning type and optional checkpoint/adapters.\n"
        "  Why: controls whether LoRA/checkpoint weights are loaded.\n"
        "  How: choose the method first, then pick checkpoint path if available.\n\n"
        "- **Quantization bit / method**\n"
        "  What: quantization level and algorithm.\n"
        "  Why: changes memory usage, speed, and output quality.\n"
        "  How: keep defaults for stability, enable quantization only when needed.\n\n"
        "- **Chat template / RoPE scaling / Booster**\n"
        "  What: prompt format, context scaling, and acceleration options.\n"
        "  Why: affects response formatting, long-context behavior, and speed.\n"
        "  How: leave defaults unless you know your model requires changes."
    )


def _update_top_help(lang: str) -> "gr.Markdown":
    return gr.Markdown(value=_make_top_help(lang))


def create_top() -> dict[str, "Component"]:
    with gr.Row():
        lang = gr.Dropdown(choices=SUPPORTED_WEBUI_LANGS, value="en", scale=1)
        available_models = list(SUPPORTED_MODELS.keys()) + ["Custom"]
        model_name = gr.Dropdown(choices=available_models, value=None, scale=2)
        model_path = gr.Textbox(scale=2)
        default_hub = "modelscope" if use_modelscope() else "openmind" if use_openmind() else "huggingface"
        hub_name = gr.Dropdown(choices=["huggingface", "modelscope", "openmind"], value=default_hub, scale=2)

    with gr.Row():
        finetuning_type = gr.Dropdown(choices=METHODS, value="lora", scale=1)
        checkpoint_path = gr.Dropdown(multiselect=True, allow_custom_value=True, scale=6)

    with gr.Row():
        quantization_bit = gr.Dropdown(choices=["none", "8", "4"], value="none", allow_custom_value=True)
        quantization_method = gr.Dropdown(choices=["bnb", "hqq", "eetq"], value="bnb")
        template = gr.Dropdown(choices=list(TEMPLATES.keys()), value="default")
        rope_scaling = gr.Dropdown(choices=["none", "linear", "dynamic", "yarn", "llama3"], value="none")
        booster = gr.Dropdown(choices=["auto", "flashattn2", "unsloth", "liger_kernel"], value="auto")

    with gr.Accordion("Top controls help", open=False, visible=False) as top_help_tab:
        top_help = gr.Markdown(value=_make_top_help("en"))

    model_name.change(get_model_info, [model_name], [model_path, template], queue=False).then(
        list_checkpoints, [model_name, finetuning_type], [checkpoint_path], queue=False
    ).then(check_template, [lang, template])
    model_name.input(save_config, inputs=[lang, hub_name, model_name], queue=False)
    model_path.input(save_config, inputs=[lang, hub_name, model_name, model_path], queue=False)
    finetuning_type.change(can_quantize, [finetuning_type], [quantization_bit], queue=False).then(
        list_checkpoints, [model_name, finetuning_type], [checkpoint_path], queue=False
    )
    checkpoint_path.focus(list_checkpoints, [model_name, finetuning_type], [checkpoint_path], queue=False)
    quantization_method.change(can_quantize_to, [quantization_method], [quantization_bit], queue=False)
    hub_name.change(switch_hub, inputs=[hub_name], queue=False).then(
        get_model_info, [model_name], [model_path, template], queue=False
    ).then(list_checkpoints, [model_name, finetuning_type], [checkpoint_path], queue=False).then(
        check_template, [lang, template]
    )
    lang.change(_update_top_help, [lang], [top_help], queue=False)
    hub_name.input(save_config, inputs=[lang, hub_name], queue=False)

    return dict(
        lang=lang,
        model_name=model_name,
        model_path=model_path,
        hub_name=hub_name,
        finetuning_type=finetuning_type,
        checkpoint_path=checkpoint_path,
        quantization_bit=quantization_bit,
        quantization_method=quantization_method,
        template=template,
        rope_scaling=rope_scaling,
        booster=booster,
        top_help_tab=top_help_tab,
        top_help=top_help,
    )
