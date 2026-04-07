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

from ...extras.packages import is_gradio_available
from ..common import is_multimodal
from ..control import auto_optimize_infer_settings
from .chatbot import create_chat_box


if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component

    from ..engine import Engine


def _make_infer_help(lang: str) -> str:
    if lang == "zh":
        return (
            "### 聊天推理帮助（简单版）\n\n"
            "- **Auto tune hardware**：自动按你的硬件选择更合适配置。\n"
            "  为什么：减少手动试错。\n"
            "  怎么用：大多数情况保持开启。\n\n"
            "- **Preferred policy**：优先策略（快 / 平衡 / 质量）。\n"
            "  为什么：决定系统偏向速度还是质量。\n"
            "  怎么用：不确定时用 `balanced`。\n\n"
            "- **Infer backend**：推理后端（huggingface / vllm / sglang / ktransformers）。\n"
            "  为什么：不同后端速度和兼容性不同。\n"
            "  怎么用：先用 `huggingface`，稳定后再尝试更快后端。\n\n"
            "- **Infer dtype**：精度类型。\n"
            "  为什么：影响显存占用和速度。\n"
            "  怎么用：一般 `auto` 即可。\n\n"
            "- **AD coordinator**：复杂任务协调器。\n"
            "  为什么：难题可分解成规划/编码/推理流程。\n"
            "  怎么用：简单对话关闭，复杂任务开启。\n\n"
            "- **Load / Unload model**：加载或释放模型。\n"
            "  为什么：节省显存与系统资源。\n"
            "  怎么用：开始聊天前先 Load，不用时 Unload。"
        )

    return (
        "### Chat Inference Help (simple)\n\n"
        "- **Auto tune hardware**: auto-picks safer settings for your machine.\n"
        "  Why: reduces manual trial-and-error.\n"
        "  How: keep enabled for most cases.\n\n"
        "- **Preferred policy**: speed vs quality preference.\n"
        "  Why: controls optimization target.\n"
        "  How: use `balanced` unless you have a clear reason to change.\n\n"
        "- **Infer backend**: runtime backend (`huggingface`, `vllm`, `sglang`, `ktransformers`).\n"
        "  Why: each backend has different compatibility/performance.\n"
        "  How: start with `huggingface`, then test faster backends later.\n\n"
        "- **Infer dtype**: numerical precision mode.\n"
        "  Why: affects memory usage and speed.\n"
        "  How: keep `auto` unless you are tuning memory manually.\n\n"
        "- **AD coordinator**: advanced multi-step coordinator for harder tasks.\n"
        "  Why: can improve complex reasoning workflows.\n"
        "  How: keep off for normal chat, enable for complex tasks.\n\n"
        "- **Load / Unload model**: starts or frees the model runtime.\n"
        "  Why: controls memory/resource usage.\n"
        "  How: Load before chatting, Unload when done."
    )


def _update_infer_help(lang: str) -> "gr.Markdown":
    return gr.Markdown(value=_make_infer_help(lang))


def create_infer_tab(engine: "Engine") -> dict[str, "Component"]:
    input_elems = engine.manager.get_base_elems()
    elem_dict = dict()

    with gr.Row():
        auto_tune_hardware = gr.Checkbox(value=True)
        auto_tune_preferred_policy = gr.Dropdown(choices=["fast", "balanced", "quality"], value="balanced")
        auto_tune_calibrate_backends = gr.Checkbox(value=False)
        auto_tune_calibration_tokens = gr.Slider(minimum=1, maximum=32, value=4, step=1)

    with gr.Row():
        infer_backend = gr.Dropdown(choices=["huggingface", "vllm", "sglang", "ktransformers"], value="huggingface")
        infer_dtype = gr.Dropdown(choices=["auto", "float16", "bfloat16", "float32"], value="auto")
        extra_args = gr.Textbox(value='{"vllm_enforce_eager": true}')

    with gr.Row():
        use_ad_coordinator = gr.Checkbox(value=False)
        ad_coordinator_policy = gr.Dropdown(choices=["fast", "balanced", "quality"], value="balanced")
        ad_complexity_threshold = gr.Slider(minimum=100, maximum=2000, value=400, step=50)

    with gr.Row():
        ad_use_role_specific_models = gr.Checkbox(value=False)
        ad_planner_model_name_or_path = gr.Textbox(value="", lines=1)
        ad_coder_model_name_or_path = gr.Textbox(value="", lines=1)
        ad_logician_model_name_or_path = gr.Textbox(value="", lines=1)

    with gr.Row():
        auto_optimize_btn = gr.Button()
        load_btn = gr.Button()
        unload_btn = gr.Button()

    with gr.Accordion("Inference help", open=False, visible=False) as infer_help_tab:
        infer_help = gr.Markdown(value=_make_infer_help("en"))

    info_box = gr.Textbox(show_label=False, interactive=False)

    input_elems.update(
        {
            infer_backend,
            infer_dtype,
            extra_args,
            auto_tune_hardware,
            auto_tune_preferred_policy,
            auto_tune_calibrate_backends,
            auto_tune_calibration_tokens,
            use_ad_coordinator,
            ad_coordinator_policy,
            ad_complexity_threshold,
            ad_use_role_specific_models,
            ad_planner_model_name_or_path,
            ad_coder_model_name_or_path,
            ad_logician_model_name_or_path,
        }
    )
    elem_dict.update(
        dict(
            infer_backend=infer_backend,
            infer_dtype=infer_dtype,
            extra_args=extra_args,
            auto_tune_hardware=auto_tune_hardware,
            auto_tune_preferred_policy=auto_tune_preferred_policy,
            auto_tune_calibrate_backends=auto_tune_calibrate_backends,
            auto_tune_calibration_tokens=auto_tune_calibration_tokens,
            use_ad_coordinator=use_ad_coordinator,
            ad_coordinator_policy=ad_coordinator_policy,
            ad_complexity_threshold=ad_complexity_threshold,
            ad_use_role_specific_models=ad_use_role_specific_models,
            ad_planner_model_name_or_path=ad_planner_model_name_or_path,
            ad_coder_model_name_or_path=ad_coder_model_name_or_path,
            ad_logician_model_name_or_path=ad_logician_model_name_or_path,
            auto_optimize_btn=auto_optimize_btn,
            load_btn=load_btn,
            unload_btn=unload_btn,
            infer_help_tab=infer_help_tab,
            infer_help=infer_help,
            info_box=info_box,
        )
    )

    chatbot, messages, chat_elems = create_chat_box(engine, visible=False)
    elem_dict.update(chat_elems)

    top_lang = engine.manager.get_elem_by_id("top.lang")
    top_lang.change(_update_infer_help, [top_lang], [infer_help], queue=False)

    auto_optimize_btn.click(
        auto_optimize_infer_settings,
        inputs=[
            engine.manager.get_elem_by_id("top.model_path"),
            engine.manager.get_elem_by_id("top.quantization_method"),
            auto_tune_preferred_policy,
            ad_use_role_specific_models,
            ad_planner_model_name_or_path,
            ad_coder_model_name_or_path,
            ad_logician_model_name_or_path,
        ],
        outputs=[
            infer_backend,
            infer_dtype,
            use_ad_coordinator,
            ad_coordinator_policy,
            ad_complexity_threshold,
            engine.manager.get_elem_by_id("top.quantization_bit"),
            info_box,
        ],
        queue=False,
    )

    load_btn.click(engine.chatter.load_model, input_elems, [info_box]).then(
        lambda: gr.Column(visible=engine.chatter.loaded), outputs=[chat_elems["chat_box"]]
    )

    unload_btn.click(engine.chatter.unload_model, input_elems, [info_box]).then(
        lambda: ([], []), outputs=[chatbot, messages]
    ).then(lambda: gr.Column(visible=engine.chatter.loaded), outputs=[chat_elems["chat_box"]])

    engine.manager.get_elem_by_id("top.model_name").change(
        lambda model_name: gr.Column(visible=is_multimodal(model_name)),
        [engine.manager.get_elem_by_id("top.model_name")],
        [chat_elems["mm_box"]],
    )

    return elem_dict
