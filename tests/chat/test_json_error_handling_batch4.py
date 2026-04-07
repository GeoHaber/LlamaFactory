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

import pytest

from llamafactory.train.reward_server_utils import parse_reward_response


class _MockResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def test_get_rewards_from_server_rejects_invalid_json_schema() -> None:
    with pytest.raises(ValueError, match="scores"):
        parse_reward_response(_MockResponse({"unexpected": []}))


def test_get_rewards_from_server_accepts_valid_payload() -> None:
    rewards = parse_reward_response(_MockResponse({"scores": [1.0, 2.0]}))
    assert rewards == [1.0, 2.0]
