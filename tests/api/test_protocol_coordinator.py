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

from llamafactory.api.protocol import ChatCompletionRequest


def test_chat_completion_request_accepts_coordinator_fields():
    request = ChatCompletionRequest(
        model="local-model",
        messages=[{"role": "user", "content": "review this python code"}],
        use_ad_coordinator=True,
        ad_coordinator_policy="quality",
        ad_complexity_threshold=512,
    )

    assert request.use_ad_coordinator is True
    assert request.ad_coordinator_policy == "quality"
    assert request.ad_complexity_threshold == 512


def test_chat_completion_request_defaults_allow_legacy_clients():
    request = ChatCompletionRequest(
        model="local-model",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert request.use_ad_coordinator is None
    assert request.ad_coordinator_policy is None
    assert request.ad_complexity_threshold is None
