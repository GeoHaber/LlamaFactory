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
from ..common import DEFAULT_DATA_DIR
from ..control import list_datasets
from .data import create_preview_box


if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component

    from ..engine import Engine


def _make_eval_help(lang: str) -> str:
    if lang == "zh":
        return (
            "### 评估菜单帮助（简单版）\n\n"
            "- **dataset_dir / dataset**：选择评估数据。\n"
            "  为什么：决定评估任务与样本。\n"
            "  怎么用：先确认 `dataset_info.json` 已配置。\n\n"
            "- **cutoff_len / max_samples / batch_size**：控制评估规模。\n"
            "  为什么：影响显存占用与速度。\n"
            "  怎么用：显存不足时降低 batch_size 或 cutoff_len。\n\n"
            "- **predict**：是否同时生成预测结果。\n"
            "  为什么：可只算指标，也可保存输出文本。\n"
            "  怎么用：做样例分析时建议开启。\n\n"
            "- **max_new_tokens / top_p / temperature**：生成参数。\n"
            "  为什么：影响输出长度和随机性。\n"
            "  怎么用：想更稳定就降低 temperature。\n\n"
            "- **output_dir**：评估结果目录。\n"
            "  为什么：保存日志、指标和预测文件。\n"
            "  怎么用：每次评估用不同目录便于对比。"
        )

    return (
        "### Evaluation Help (simple)\n\n"
        "- **dataset_dir / dataset**: choose evaluation data source.\n"
        "  Why: determines what task and samples are evaluated.\n"
        "  How: make sure your dataset exists in `dataset_info.json`.\n\n"
        "- **cutoff_len / max_samples / batch_size**: evaluation size controls.\n"
        "  Why: they control memory usage and runtime.\n"
        "  How: reduce batch size or cutoff length if you hit OOM.\n\n"
        "- **predict**: generate predictions during evaluation.\n"
        "  Why: useful when you want metrics and sample outputs together.\n"
        "  How: disable for metrics-only runs, enable for output inspection.\n\n"
        "- **max_new_tokens / top_p / temperature**: generation behavior.\n"
        "  Why: affects output length and randomness.\n"
        "  How: lower temperature for more stable outputs.\n\n"
        "- **output_dir**: folder for evaluation artifacts.\n"
        "  Why: stores logs, metrics, and optional predictions.\n"
        "  How: use a unique folder per run for easier comparisons."
    )


def _update_eval_help(lang: str) -> "gr.Markdown":
    return gr.Markdown(value=_make_eval_help(lang))


def create_eval_tab(engine: "Engine") -> dict[str, "Component"]:
    input_elems = engine.manager.get_base_elems()
    elem_dict = dict()

    with gr.Row():
        dataset_dir = gr.Textbox(value=DEFAULT_DATA_DIR, scale=2)
        dataset = gr.Dropdown(multiselect=True, allow_custom_value=True, scale=4)
        preview_elems = create_preview_box(dataset_dir, dataset)

    input_elems.update({dataset_dir, dataset})
    elem_dict.update(dict(dataset_dir=dataset_dir, dataset=dataset, **preview_elems))

    with gr.Row():
        cutoff_len = gr.Slider(minimum=4, maximum=131072, value=1024, step=1)
        max_samples = gr.Textbox(value="100000")
        batch_size = gr.Slider(minimum=1, maximum=1024, value=2, step=1)
        predict = gr.Checkbox(value=True)

    input_elems.update({cutoff_len, max_samples, batch_size, predict})
    elem_dict.update(dict(cutoff_len=cutoff_len, max_samples=max_samples, batch_size=batch_size, predict=predict))

    with gr.Row():
        max_new_tokens = gr.Slider(minimum=8, maximum=4096, value=512, step=1)
        top_p = gr.Slider(minimum=0.01, maximum=1, value=0.7, step=0.01)
        temperature = gr.Slider(minimum=0.01, maximum=1.5, value=0.95, step=0.01)
        output_dir = gr.Textbox()

    with gr.Accordion("Evaluation help", open=False, visible=False) as eval_help_tab:
        eval_help = gr.Markdown(value=_make_eval_help("en"))

    input_elems.update({max_new_tokens, top_p, temperature, output_dir})
    elem_dict.update(dict(max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature, output_dir=output_dir))

    with gr.Row():
        cmd_preview_btn = gr.Button()
        start_btn = gr.Button(variant="primary")
        stop_btn = gr.Button(variant="stop")

    with gr.Row():
        resume_btn = gr.Checkbox(visible=False, interactive=False)
        progress_bar = gr.Slider(visible=False, interactive=False)

    with gr.Row():
        output_box = gr.Markdown()

    elem_dict.update(
        dict(
            cmd_preview_btn=cmd_preview_btn,
            start_btn=start_btn,
            stop_btn=stop_btn,
            resume_btn=resume_btn,
            progress_bar=progress_bar,
            eval_help_tab=eval_help_tab,
            eval_help=eval_help,
            output_box=output_box,
        )
    )
    output_elems = [output_box, progress_bar]

    cmd_preview_btn.click(engine.runner.preview_eval, input_elems, output_elems, concurrency_limit=None)
    start_btn.click(engine.runner.run_eval, input_elems, output_elems)
    stop_btn.click(engine.runner.set_abort)
    resume_btn.change(engine.runner.monitor, outputs=output_elems, concurrency_limit=None)
    top_lang = engine.manager.get_elem_by_id("top.lang")
    top_lang.change(_update_eval_help, [top_lang], [eval_help], queue=False)

    dataset.focus(list_datasets, [dataset_dir], [dataset], queue=False)

    return elem_dict
