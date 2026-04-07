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
from typing import Any


def load_data_file(file_path: str) -> list[Any]:
    with open(file_path, encoding="utf-8") as f:
        if file_path.endswith(".json"):
            try:
                payload = json.load(f)
            except json.JSONDecodeError:
                return []

            return payload if isinstance(payload, list) else [payload]
        elif file_path.endswith(".jsonl"):
            rows: list[Any] = []
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

            return rows
        else:
            return list(f)