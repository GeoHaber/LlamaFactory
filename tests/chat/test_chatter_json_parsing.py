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

from llamafactory.webui.json_utils import loads_json_dict


def test_loads_json_dict_returns_dict() -> None:
    parsed = loads_json_dict('{"max_new_tokens": 128, "temperature": 0.7}')
    assert parsed == {"max_new_tokens": 128, "temperature": 0.7}


def test_loads_json_dict_rejects_invalid_json() -> None:
    assert loads_json_dict('{"max_new_tokens": 128') is None


def test_loads_json_dict_rejects_non_object_payload() -> None:
    assert loads_json_dict('[1, 2, 3]') is None
