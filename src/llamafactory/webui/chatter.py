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
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from transformers.utils import is_torch_npu_available

from ..chat import ChatModel
from ..chat.autotune import HardwareAutoTuner
from ..data import Role
from ..extras.constants import PEFT_METHODS
from ..extras.misc import torch_gc
from ..extras.packages import is_gradio_available, is_kt_available, is_sglang_available, is_vllm_available
from .common import get_save_dir, load_config
from .json_utils import loads_json_dict
from .locales import ALERTS


if TYPE_CHECKING:
    from ..chat import BaseEngine
    from .manager import Manager


if is_gradio_available():
    import gradio as gr


def _escape_html(text: str) -> str:
    r"""Escape HTML characters."""
    return text.replace("<", "&lt;").replace(">", "&gt;")


def _format_response(text: str, lang: str, escape_html: bool, thought_words: tuple[str, str]) -> str:
    r"""Post-process the response text.

    Based on: https://huggingface.co/spaces/Lyte/DeepSeek-R1-Distill-Qwen-1.5B-Demo-GGUF/blob/main/app.py
    """
    if thought_words[0] not in text:
        return _escape_html(text) if escape_html else text

    text = text.replace(thought_words[0], "")
    result = text.split(thought_words[1], maxsplit=1)
    if len(result) == 1:
        summary = ALERTS["info_thinking"][lang]
        thought, answer = text, ""
    else:
        summary = ALERTS["info_thought"][lang]
        thought, answer = result

    if escape_html:
        thought, answer = _escape_html(thought), _escape_html(answer)

    return (
        f"<details open><summary class='thinking-summary'><span>{summary}</span></summary>\n\n"
        f"<div class='thinking-container'>\n{thought}\n</div>\n</details>{answer}"
    )


@contextmanager
def update_attr(obj: Any, name: str, value: Any):
    old_value = getattr(obj, name, None)
    setattr(obj, name, value)
    yield
    setattr(obj, name, old_value)


class WebChatModel(ChatModel):
    def __init__(self, manager: "Manager", demo_mode: bool = False, lazy_init: bool = True) -> None:
        self.manager = manager
        self.demo_mode = demo_mode
        self.engine: BaseEngine | None = None

        if not lazy_init:  # read arguments from command line
            super().__init__()

        if demo_mode and os.getenv("DEMO_MODEL") and os.getenv("DEMO_TEMPLATE"):  # load demo model
            model_name_or_path = os.getenv("DEMO_MODEL")
            template = os.getenv("DEMO_TEMPLATE")
            infer_backend = os.getenv("DEMO_BACKEND", "huggingface")
            super().__init__(
                dict(model_name_or_path=model_name_or_path, template=template, infer_backend=infer_backend)
            )

    @property
    def loaded(self) -> bool:
        return self.engine is not None

    @staticmethod
    def _calibrate_backend(base_args: dict[str, Any], max_new_tokens: int) -> tuple[str | None, dict[str, float]]:
        timings: dict[str, float] = {}
        candidates = ["huggingface"]
        if is_vllm_available():
            candidates.append("vllm")
        if is_sglang_available():
            candidates.append("sglang")
        if is_kt_available():
            candidates.append("ktransformers")

        for backend in candidates:
            trial_model = None
            try:
                trial_args = dict(base_args)
                trial_args["infer_backend"] = backend
                trial_args["use_ad_coordinator"] = False
                trial_args["ad_use_role_specific_models"] = False
                trial_model = ChatModel(trial_args)

                t0 = time.perf_counter()
                trial_model.chat(
                    [{"role": "user", "content": "Reply with OK."}],
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.1,
                    top_p=1.0,
                )
                timings[backend] = (time.perf_counter() - t0) * 1000.0
            except Exception:
                continue
            finally:
                if trial_model is not None:
                    trial_model.engine = None
                torch_gc()

        if not timings:
            return None, timings

        best_backend = min(timings, key=timings.get)
        return best_backend, timings

    def load_model(self, data) -> Generator[str, None, None]:
        get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]
        lang, model_name, model_path = get("top.lang"), get("top.model_name"), get("top.model_path")
        finetuning_type, checkpoint_path = get("top.finetuning_type"), get("top.checkpoint_path")
        user_config = load_config()

        error = ""
        if self.loaded:
            error = ALERTS["err_exists"][lang]
        elif not model_name:
            error = ALERTS["err_no_model"][lang]
        elif not model_path:
            error = ALERTS["err_no_path"][lang]
        elif self.demo_mode:
            error = ALERTS["err_demo"][lang]

        extra_args = loads_json_dict(get("infer.extra_args"))
        if extra_args is None:
            error = ALERTS["err_json_schema"][lang]

        if error:
            gr.Warning(error)
            yield error
            return

        yield ALERTS["info_loading"][lang]

        recommendation = None
        if get("infer.auto_tune_hardware"):
            recommendation = HardwareAutoTuner.recommend(get("infer.auto_tune_preferred_policy"))
            os.environ["MAX_CONCURRENT"] = str(recommendation.max_concurrent)

        args = dict(
            model_name_or_path=model_path,
            cache_dir=user_config.get("cache_dir", None),
            finetuning_type=finetuning_type,
            template=get("top.template"),
            rope_scaling=get("top.rope_scaling") if get("top.rope_scaling") != "none" else None,
            flash_attn="fa2" if get("top.booster") == "flashattn2" else "auto",
            use_unsloth=(get("top.booster") == "unsloth"),
            enable_liger_kernel=(get("top.booster") == "liger_kernel"),
            infer_backend=(recommendation.infer_backend if recommendation else get("infer.infer_backend")),
            infer_dtype=(recommendation.infer_dtype if recommendation else get("infer.infer_dtype")),
            use_ad_coordinator=get("infer.use_ad_coordinator"),
            ad_coordinator_policy=(recommendation.ad_policy if recommendation else get("infer.ad_coordinator_policy")),
            ad_complexity_threshold=int(get("infer.ad_complexity_threshold")),
            ad_use_role_specific_models=get("infer.ad_use_role_specific_models"),
            ad_planner_model_name_or_path=(get("infer.ad_planner_model_name_or_path") or None),
            ad_coder_model_name_or_path=(get("infer.ad_coder_model_name_or_path") or None),
            ad_logician_model_name_or_path=(get("infer.ad_logician_model_name_or_path") or None),
            trust_remote_code=True,
        )
        args.update(extra_args)

        # checkpoints
        if checkpoint_path:
            if finetuning_type in PEFT_METHODS:  # list
                args["adapter_name_or_path"] = ",".join(
                    [get_save_dir(model_name, finetuning_type, adapter) for adapter in checkpoint_path]
                )
            else:  # str
                args["model_name_or_path"] = get_save_dir(model_name, finetuning_type, checkpoint_path)

        # quantization
        selected_quant_bit = get("top.quantization_bit")
        if selected_quant_bit != "none":
            args["quantization_bit"] = int(get("top.quantization_bit"))
            args["quantization_method"] = get("top.quantization_method")
            args["double_quantization"] = not is_torch_npu_available()
        elif recommendation and recommendation.quantization_bit is not None:
            args["quantization_bit"] = recommendation.quantization_bit
            args["quantization_method"] = get("top.quantization_method")
            args["double_quantization"] = not is_torch_npu_available()

        if get("infer.auto_tune_calibrate_backends"):
            best_backend, timings = self._calibrate_backend(
                base_args=args,
                max_new_tokens=int(get("infer.auto_tune_calibration_tokens")),
            )
            if best_backend is not None:
                args["infer_backend"] = best_backend
                gr.Info(f"Backend calibration selected '{best_backend}' with timings(ms): {timings}")
            else:
                gr.Warning("Backend calibration found no valid backend. Keeping configured backend.")

        if recommendation:
            gr.Info(f"Hardware auto tune: {recommendation.summary}")

        super().__init__(args)
        yield ALERTS["info_loaded"][lang]

    def unload_model(self, data) -> Generator[str, None, None]:
        lang = data[self.manager.get_elem_by_id("top.lang")]

        if self.demo_mode:
            gr.Warning(ALERTS["err_demo"][lang])
            yield ALERTS["err_demo"][lang]
            return

        yield ALERTS["info_unloading"][lang]
        self.engine = None
        torch_gc()
        yield ALERTS["info_unloaded"][lang]

    @staticmethod
    def append(
        chatbot: list[dict[str, str]],
        messages: list[dict[str, str]],
        role: str,
        query: str,
        escape_html: bool,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]], str]:
        r"""Add the user input to chatbot.

        Inputs: infer.chatbot, infer.messages, infer.role, infer.query, infer.escape_html
        Output: infer.chatbot, infer.messages, infer.query
        """
        return (
            chatbot + [{"role": "user", "content": _escape_html(query) if escape_html else query}],
            messages + [{"role": role, "content": query}],
            "",
        )

    def stream(
        self,
        chatbot: list[dict[str, str]],
        messages: list[dict[str, str]],
        lang: str,
        system: str,
        tools: str,
        image: Any | None,
        video: Any | None,
        audio: Any | None,
        max_new_tokens: int,
        top_p: float,
        temperature: float,
        skip_special_tokens: bool,
        escape_html: bool,
        enable_thinking: bool,
    ) -> Generator[tuple[list[dict[str, str]], list[dict[str, str]], str, str, str], None, None]:
        r"""Generate output text in stream.

        Inputs: infer.chatbot, infer.messages, infer.system, infer.tools, infer.image, infer.video, ...
        Output: infer.chatbot, infer.messages
        """
        with update_attr(self.engine.template, "enable_thinking", enable_thinking):
            chatbot.append({"role": "assistant", "content": ""})
            response = ""
            for new_text in self.stream_chat(
                messages,
                system,
                tools,
                images=[image] if image else None,
                videos=[video] if video else None,
                audios=[audio] if audio else None,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                temperature=temperature,
                skip_special_tokens=skip_special_tokens,
            ):
                response += new_text
                if tools:
                    result = self.engine.template.extract_tool(response)
                else:
                    result = response

                if isinstance(result, list):
                    tool_calls = []
                    for tool in result:
                        arguments = loads_json_dict(tool.arguments)
                        if arguments is None:
                            arguments = {"_raw": tool.arguments, "_error": "invalid_json"}

                        tool_calls.append({"name": tool.name, "arguments": arguments})

                    tool_calls = json.dumps(tool_calls, ensure_ascii=False)
                    output_messages = messages + [{"role": Role.FUNCTION.value, "content": tool_calls}]
                    bot_text = "```json\n" + tool_calls + "\n```"
                else:
                    output_messages = messages + [{"role": Role.ASSISTANT.value, "content": result}]
                    bot_text = _format_response(result, lang, escape_html, self.engine.template.thought_words)

                chatbot[-1] = {"role": "assistant", "content": bot_text}
                planner_trace = ""
                coder_trace = ""
                logician_trace = ""
                if self.coordinator is not None:
                    trace = self.coordinator.last_trace
                    planner_trace = trace.planner
                    coder_trace = trace.coder
                    logician_trace = trace.logician

                yield chatbot, output_messages, planner_trace, coder_trace, logician_trace
