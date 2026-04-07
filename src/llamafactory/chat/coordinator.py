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

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Optional

from .base_engine import Response


CoordinatorChatCallable = Callable[..., Awaitable[list[Response]]]


@dataclass
class CoordinatorTrace:
    policy: str
    is_complex: bool
    planner: str = ""
    coder: str = ""
    logician: str = ""
    final: str = ""


class ADCoordinator:
    r"""Adaptive decomposition coordinator for coding-oriented reasoning workflows.

    This first version reuses the currently loaded local model for all roles and
    varies prompts and call sequence by policy. It is intentionally lightweight
    and non-breaking, and can be extended to per-role models later.
    """

    def __init__(
        self,
        chat_callable: CoordinatorChatCallable,
        role_chat_callables: Optional[dict[str, CoordinatorChatCallable]] = None,
        policy: str = "balanced",
        complexity_threshold: int = 400,
    ) -> None:
        self._chat = chat_callable
        self._role_chat_callables = role_chat_callables or {}
        self.policy = policy
        self.complexity_threshold = complexity_threshold
        self.last_trace = CoordinatorTrace(policy=policy, is_complex=False)

    @staticmethod
    def _role_prompt(role: str) -> str:
        prompts = {
            "planner": (
                "You are a planning specialist. Produce a compact step-by-step plan, "
                "highlight assumptions, and call out risky steps before implementation."
            ),
            "coder": (
                "You are a senior coding specialist. Produce precise, minimal, and correct code changes. "
                "Prioritize correctness and maintainability."
            ),
            "logician": (
                "You are a critical reviewer. Stress-test edge cases, detect logical flaws, and "
                "propose corrections with concise rationale."
            ),
        }
        return prompts[role]

    @staticmethod
    def _last_user_text(messages: list[dict[str, str]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                return message.get("content", "")

        return ""

    def _is_complex(self, messages: list[dict[str, str]]) -> bool:
        query = self._last_user_text(messages).lower()
        if len(query) >= self.complexity_threshold:
            return True

        complexity_terms = (
            "refactor",
            "architecture",
            "translate",
            "rust",
            "performance",
            "distill",
            "multi",
            "coordinator",
            "benchmark",
            "security",
        )
        return any(term in query for term in complexity_terms)

    async def _call_role(
        self,
        role: str,
        messages: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        **input_kwargs: Any,
    ) -> Response:
        role_system = self._role_prompt(role)
        merged_system = role_system if not system else f"{system}\n\n{role_system}"
        role_chat = self._role_chat_callables.get(role, self._chat)
        outputs = await role_chat(messages, merged_system, tools, **input_kwargs)
        return outputs[0]

    async def chat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        **input_kwargs: Any,
    ) -> list[Response]:
        complex_query = self._is_complex(messages)
        trace = CoordinatorTrace(policy=self.policy, is_complex=complex_query)

        if self.policy == "fast":
            coder = await self._call_role("coder", messages, system, tools, **input_kwargs)
            trace.coder = coder.response_text
            trace.final = coder.response_text
            self.last_trace = trace
            return [coder]

        planner = await self._call_role("planner", messages, system, tools, **input_kwargs)
        trace.planner = planner.response_text

        if self.policy == "balanced":
            coder = await self._call_role("coder", messages, system, tools, **input_kwargs)
            trace.coder = coder.response_text
            if not complex_query:
                trace.final = coder.response_text
                self.last_trace = trace
                return [coder]

            critique = await self._call_role("logician", messages, system, tools, **input_kwargs)
            trace.logician = critique.response_text
            final_text = f"{planner.response_text}\n\n{coder.response_text}\n\n{critique.response_text}"
            trace.final = final_text
            self.last_trace = trace
            return [
                Response(
                    response_text=final_text,
                    response_length=len(final_text),
                    prompt_length=max(planner.prompt_length, coder.prompt_length, critique.prompt_length),
                    finish_reason="stop",
                )
            ]

        # quality mode: planner -> coder -> logician critique -> coder refinement
        first_coder = await self._call_role("coder", messages, system, tools, **input_kwargs)
        trace.coder = first_coder.response_text
        critique = await self._call_role("logician", messages, system, tools, **input_kwargs)
        trace.logician = critique.response_text

        refinement_messages = messages + [
            {
                "role": "assistant",
                "content": (
                    "Initial plan:\n"
                    + planner.response_text
                    + "\n\nInitial draft:\n"
                    + first_coder.response_text
                    + "\n\nCritical review:\n"
                    + critique.response_text
                    + "\n\nNow provide the final improved answer."
                ),
            }
        ]
        refined = await self._call_role("coder", refinement_messages, system, tools, **input_kwargs)
        trace.coder = f"Initial draft:\n{first_coder.response_text}\n\nRefined:\n{refined.response_text}"
        trace.final = refined.response_text
        self.last_trace = trace
        return [refined]
