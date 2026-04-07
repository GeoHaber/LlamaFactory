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

from pathlib import Path

from llamafactory.webui.common import load_eval_results
from llamafactory.webui.data_utils import load_data_file


def test_load_data_file_ignores_bad_jsonl_rows(tmp_path: Path) -> None:
    sample = tmp_path / "sample.jsonl"
    sample.write_text('{"ok": 1}\n{broken\n{"ok": 2}\n', encoding="utf-8")

    rows = load_data_file(str(sample))
    assert rows == [{"ok": 1}, {"ok": 2}]


def test_load_data_file_returns_empty_for_malformed_json(tmp_path: Path) -> None:
    sample = tmp_path / "sample.json"
    sample.write_text('{broken', encoding="utf-8")

    rows = load_data_file(str(sample))
    assert rows == []


def test_load_eval_results_handles_malformed_json(tmp_path: Path) -> None:
    sample = tmp_path / "result.json"
    sample.write_text('{broken', encoding="utf-8")

    rendered = load_eval_results(sample)
    assert "Malformed JSON" in rendered
