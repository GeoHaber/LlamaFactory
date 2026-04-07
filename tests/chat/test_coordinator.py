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

import asyncio
from typing import Any, Optional

from llamafactory.chat import ADCoordinator
from llamafactory.chat.base_engine import Response


class FakeChatBackend:
    def __init__(self):
        self.system_prompts: list[str] = []

    async def chat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        **kwargs: Any,
    ) -> list[Response]:
        del messages, tools, kwargs
        self.system_prompts.append(system or "")
        role = "coder"
        if "planning specialist" in (system or ""):
            role = "planner"
        elif "critical reviewer" in (system or ""):
            role = "logician"

        return [Response(response_text=f"{role}-out", response_length=8, prompt_length=10, finish_reason="stop")]


def test_fast_policy_calls_only_coder():
    backend = FakeChatBackend()
    coordinator = ADCoordinator(backend.chat, policy="fast")

    result = asyncio.run(coordinator.chat([{"role": "user", "content": "quick answer"}]))

    assert len(backend.system_prompts) == 1
    assert "coding specialist" in backend.system_prompts[0]
    assert result[0].response_text == "coder-out"


def test_balanced_policy_simple_query_skips_logician():
    backend = FakeChatBackend()
    coordinator = ADCoordinator(backend.chat, policy="balanced", complexity_threshold=100)

    result = asyncio.run(coordinator.chat([{"role": "user", "content": "small task"}]))

    assert len(backend.system_prompts) == 2
    assert "planning specialist" in backend.system_prompts[0]
    assert "coding specialist" in backend.system_prompts[1]
    assert result[0].response_text == "coder-out"


def test_balanced_policy_complex_query_calls_logician():
    backend = FakeChatBackend()
    coordinator = ADCoordinator(backend.chat, policy="balanced", complexity_threshold=10)

    result = asyncio.run(
        coordinator.chat([{"role": "user", "content": "translate this python architecture to rust"}])
    )

    assert len(backend.system_prompts) == 3
    assert "planning specialist" in backend.system_prompts[0]
    assert "coding specialist" in backend.system_prompts[1]
    assert "critical reviewer" in backend.system_prompts[2]
    assert "planner-out" in result[0].response_text
    assert "coder-out" in result[0].response_text
    assert "logician-out" in result[0].response_text


def test_quality_policy_runs_refinement_round():
    backend = FakeChatBackend()
    coordinator = ADCoordinator(backend.chat, policy="quality")

    result = asyncio.run(coordinator.chat([{"role": "user", "content": "improve this"}]))

    assert len(backend.system_prompts) == 4
    assert "planning specialist" in backend.system_prompts[0]
    assert "coding specialist" in backend.system_prompts[1]
    assert "critical reviewer" in backend.system_prompts[2]
    assert "coding specialist" in backend.system_prompts[3]
    assert result[0].response_text == "coder-out"


def test_role_specific_callables_are_used():
    backend = FakeChatBackend()
    planner_backend = FakeChatBackend()

    coordinator = ADCoordinator(
        backend.chat,
        role_chat_callables={"planner": planner_backend.chat},
        policy="balanced",
        complexity_threshold=100,
    )

    _ = asyncio.run(coordinator.chat([{"role": "user", "content": "small task"}]))

    assert len(planner_backend.system_prompts) == 1
    assert "planning specialist" in planner_backend.system_prompts[0]
    assert len(backend.system_prompts) == 1
    assert "coding specialist" in backend.system_prompts[0]
