#!/usr/bin/env python
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

"""Pipeline Event Logger — structured JSON event logging for distillation scripts.

Provides a simple event logger that writes structured JSON lines alongside
normal print output. Each event has a timestamp, level, stage, and payload.

Usage in pipeline scripts:
    from pipeline_events import PipelineLogger

    log = PipelineLogger("purification", "data/zena007/events.jsonl")
    log.info("started", total_samples=1200)
    log.info("classified", sample_id="tr-001", tier="GOLD")
    log.warn("low_agreement", sample_id="tr-099", agreement=0.45)
    log.error("crash", error="OOM", traceback="...")
    log.close()
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


class PipelineLogger:
    """Structured event logger for pipeline scripts.

    Writes JSON events to a log file and optionally prints to stderr.
    """

    def __init__(
        self,
        stage: str,
        log_path: str | Path | None = None,
        echo: bool = True,
    ) -> None:
        self.stage = stage
        self.echo = echo
        self._start = time.time()
        self._file = None
        if log_path:
            p = Path(log_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._file = p.open("a", encoding="utf-8")

    def _emit(self, level: str, event: str, **kwargs: object) -> None:
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_s": round(time.time() - self._start, 2),
            "stage": self.stage,
            "level": level,
            "event": event,
        }
        record.update(kwargs)
        line = json.dumps(record, ensure_ascii=False, default=str)
        if self._file:
            self._file.write(line + "\n")
            self._file.flush()
        if self.echo:
            prefix = {"INFO": "·", "WARN": "⚠", "ERROR": "✗"}.get(level, "·")
            print(f"  {prefix} [{self.stage}] {event}", file=sys.stderr, flush=True)  # xray: ignore[PY-004]

    def info(self, event: str, **kwargs: object) -> None:
        self._emit("INFO", event, **kwargs)

    def warn(self, event: str, **kwargs: object) -> None:
        self._emit("WARN", event, **kwargs)

    def error(self, event: str, **kwargs: object) -> None:
        self._emit("ERROR", event, **kwargs)

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None

    def __enter__(self) -> PipelineLogger:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def load_events(log_path: str | Path) -> list[dict]:
    """Load events from a JSONL event log file."""
    path = Path(log_path)
    if not path.exists():
        return []
    events: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                events.append(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                pass
    return events


def summarize_events(events: list[dict]) -> dict:
    """Summarize events by stage and level."""
    summary: dict[str, dict[str, int]] = {}
    for e in events:
        stage = e.get("stage", "unknown")
        level = e.get("level", "INFO")
        if stage not in summary:
            summary[stage] = {"INFO": 0, "WARN": 0, "ERROR": 0}
        summary[stage][level] = summary[stage].get(level, 0) + 1
    return summary
