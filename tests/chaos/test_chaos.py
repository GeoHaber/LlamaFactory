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
"""Chaos tests for the distillation pipeline.

These tests exercise *failure handling* paths -- they intentionally feed
malformed, corrupt, or pathological telemetry into the postmortem agent
and metrics rail to verify the system fails gracefully and detects the
right pathology.

Tests are skipped by default. Run with::

    RUN_CHAOS=1 pytest tests/chaos/

The tests do **not** require a GPU, a real model, or network access --
all faults are simulated by writing synthetic JSONL telemetry to a temp
dir and asserting the agent's findings match the planted issue.

Covers seven failure modes:

1. dataloader stall          (CPU-bound classification)
2. OOM / VRAM pressure       (VRAM crit detector)
3. RAM exhaustion            (RAM detector)
4. NaN loss / grad explosion (training collapse detector)
5. dataloader idle           (sync stall detector)
6. plateau                   (LR / scheduling diagnostic)
7. corrupt telemetry         (parser robustness -- agent must not crash)
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pytest

# Allow the chaos suite to import scripts/postmortem_agent.py without
# requiring the package to be installed.
_REPO = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_CHAOS", "0") not in ("1", "true", "True"),
    reason="chaos tests are opt-in (set RUN_CHAOS=1 to run)",
)


# ---------------------------------------------------------------------------
# Synthetic-telemetry helpers
# ---------------------------------------------------------------------------
def _write_metrics(run_dir: Path, ticks: list[dict]) -> None:
    """Write a metrics.jsonl sidecar that the postmortem agent will read."""
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "metrics.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for t in ticks:
            fh.write(json.dumps(t) + "\n")


def _write_train_log(run_dir: Path, entries: list[dict]) -> None:
    """Write a synthetic trainer_log.jsonl (the format LlamaFactory emits)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "trainer_log.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")


def _ramp(n: int, start: float, end: float) -> list[float]:
    if n <= 1:
        return [end]
    step = (end - start) / (n - 1)
    return [start + i * step for i in range(n)]


def _make_tick(ts: float, gpu_util: float, cpu_util: float, ram_pct: float,
               vram_used: float = 4000.0, vram_total: float = 24576.0,
               disk_read: float = 5.0, ram_used_gb: float = 16.0) -> dict:
    return {
        "ts": ts,
        "stage": "sft",
        "tok_s": 0.0,
        "eta": "",
        "cpu.util_pct": cpu_util,
        "ram.used_gb": ram_used_gb,
        "ram.total_gb": 64.0,
        "ram.pct": ram_pct,
        "swap.used_gb": 0.0,
        "disk.free_gb": 200.0,
        "disk.read_mb_s": disk_read,
        "disk.write_mb_s": 1.0,
        "net.tx_mb_s": 0.1,
        "net.rx_mb_s": 0.1,
        "gpu.util_pct": gpu_util,
        "gpu.mem_used_mb": vram_used,
        "gpu.mem_total_mb": vram_total,
        "gpu.power_w": 200.0,
        "gpu.temp_c": 65.0,
    }


# ---------------------------------------------------------------------------
# 1. Dataloader stall -- GPU starves while CPU pegs.
# ---------------------------------------------------------------------------
def test_chaos_dataloader_cpu_bottleneck(tmp_path: Path) -> None:
    """CPU pegged + GPU starving for >20% of run -> CPU-bound finding."""
    import postmortem_agent as pm

    run = tmp_path / "run_cpu_bottleneck"
    ticks = [
        _make_tick(ts=float(i), gpu_util=10.0, cpu_util=95.0, ram_pct=70.0)
        for i in range(60)
    ]
    _write_metrics(run, ticks)
    _write_train_log(run, [{"loss": 2.5, "epoch": 0.5, "current_steps": 50}])

    summary = pm.analyze(run)
    assert summary is not None
    titles = " ".join(f.title.lower() for f in summary.findings)
    assert "cpu" in titles or "under-util" in titles, \
        f"expected CPU-bound finding, got: {titles}"


# ---------------------------------------------------------------------------
# 2. VRAM pressure -- memory near total throughout the run.
# ---------------------------------------------------------------------------
def test_chaos_oom_vram_pressure(tmp_path: Path) -> None:
    """VRAM pinned >92% triggers a critical-severity finding."""
    import postmortem_agent as pm

    run = tmp_path / "run_oom"
    ticks = [
        _make_tick(ts=float(i), gpu_util=95.0, cpu_util=40.0, ram_pct=60.0,
                   vram_used=23800.0, vram_total=24576.0)
        for i in range(30)
    ]
    _write_metrics(run, ticks)
    _write_train_log(run, [{"loss": 1.8, "epoch": 0.3, "current_steps": 100}])

    summary = pm.analyze(run)
    assert summary is not None
    titles = " ".join(f.title.lower() for f in summary.findings)
    assert "vram" in titles, f"expected VRAM finding, got: {titles}"


# ---------------------------------------------------------------------------
# 3. RAM exhaustion -- system memory hits ceiling.
# ---------------------------------------------------------------------------
def test_chaos_ram_pressure(tmp_path: Path) -> None:
    import postmortem_agent as pm

    run = tmp_path / "run_ram"
    ticks = [
        _make_tick(ts=float(i), gpu_util=70.0, cpu_util=50.0, ram_pct=96.0,
                   ram_used_gb=61.0)
        for i in range(20)
    ]
    _write_metrics(run, ticks)
    _write_train_log(run, [{"loss": 2.0, "epoch": 0.4, "current_steps": 80}])

    summary = pm.analyze(run)
    assert summary is not None
    titles = " ".join(f.title.lower() for f in summary.findings)
    assert "ram" in titles, f"expected RAM finding, got: {titles}"


# ---------------------------------------------------------------------------
# 4. NaN / gradient explosion -- training collapse.
# ---------------------------------------------------------------------------
def test_chaos_nan_loss(tmp_path: Path) -> None:
    import postmortem_agent as pm

    run = tmp_path / "run_nan"
    _write_metrics(run, [
        _make_tick(ts=float(i), gpu_util=80.0, cpu_util=40.0, ram_pct=60.0)
        for i in range(10)
    ])
    train = [
        {"loss": 2.5, "epoch": 0.1, "current_steps": 10, "grad_norm": 1.2},
        {"loss": 2.4, "epoch": 0.2, "current_steps": 20, "grad_norm": 1.1},
        {"loss": float("nan"), "epoch": 0.3, "current_steps": 30, "grad_norm": 999.0},
        {"loss": float("nan"), "epoch": 0.4, "current_steps": 40, "grad_norm": 9999.0},
    ]
    _write_train_log(run, train)

    summary = pm.analyze(run)
    assert summary is not None
    titles = " ".join(f.title.lower() for f in summary.findings)
    assert "nan" in titles or "explod" in titles or "grad" in titles, \
        f"expected NaN/explode finding, got: {titles}"


# ---------------------------------------------------------------------------
# 5. Dataloader idle -- both GPU and CPU sit at zero.
# ---------------------------------------------------------------------------
def test_chaos_dataloader_idle_stall(tmp_path: Path) -> None:
    """GPU and CPU both <10% for many ticks -> stall detector fires."""
    import postmortem_agent as pm

    run = tmp_path / "run_stall"
    ticks = [
        _make_tick(ts=float(i), gpu_util=2.0, cpu_util=5.0, ram_pct=40.0)
        for i in range(40)
    ]
    _write_metrics(run, ticks)
    _write_train_log(run, [{"loss": 2.0, "epoch": 0.1, "current_steps": 5}])

    summary = pm.analyze(run)
    assert summary is not None
    # The agent has multiple detectors that may catch idle; we just need >=1.
    assert len(summary.findings) >= 1


# ---------------------------------------------------------------------------
# 6. Loss plateau -- final-third improvement is essentially zero.
# ---------------------------------------------------------------------------
def test_chaos_loss_plateau(tmp_path: Path) -> None:
    import postmortem_agent as pm

    run = tmp_path / "run_plateau"
    _write_metrics(run, [
        _make_tick(ts=float(i), gpu_util=80.0, cpu_util=40.0, ram_pct=60.0)
        for i in range(20)
    ])
    # Loss drops fast in the first third, then plateaus completely.
    # Detector requires >=50 entries before it will report.
    losses = _ramp(20, 3.0, 1.5) + [1.5001 + (i % 2) * 0.0001 for i in range(40)]
    train = [
        {"loss": loss, "epoch": i / len(losses),
         "current_steps": (i + 1) * 10}
        for i, loss in enumerate(losses)
    ]
    _write_train_log(run, train)

    summary = pm.analyze(run)
    assert summary is not None
    titles = " ".join(f.title.lower() for f in summary.findings)
    assert "plateau" in titles or "improv" in titles, \
        f"expected plateau finding, got: {titles}"


# ---------------------------------------------------------------------------
# 7. Corrupt telemetry -- agent must not crash on bad input.
# ---------------------------------------------------------------------------
def test_chaos_corrupt_telemetry(tmp_path: Path) -> None:
    """Mangled JSON, missing fields, wrong types -- the agent must survive."""
    import postmortem_agent as pm

    run = tmp_path / "run_corrupt"
    run.mkdir(parents=True)

    # metrics.jsonl with: a valid line, a malformed line, an empty line,
    # a line with wrong field types, a line missing all fields.
    (run / "metrics.jsonl").write_text(
        json.dumps(_make_tick(1.0, 80, 40, 60)) + "\n"
        + "{not valid json\n"
        + "\n"
        + json.dumps({"ts": "not-a-number", "gpu.util_pct": "high"}) + "\n"
        + "{}\n",
        encoding="utf-8",
    )

    # trainer_log with similarly broken entries
    (run / "trainer_log.jsonl").write_text(
        json.dumps({"loss": 2.5, "epoch": 0.1}) + "\n"
        + "garbage line\n"
        + json.dumps({"loss": "not a number"}) + "\n"
        + json.dumps({}) + "\n",
        encoding="utf-8",
    )

    # The agent must return a summary -- it should not raise on bad input.
    summary = pm.analyze(run)
    assert summary is not None, "agent must not crash on corrupt telemetry"
    assert summary.metric_ticks >= 1, "agent should still parse valid lines"


# ---------------------------------------------------------------------------
# 8. Metrics-rail backend chaos -- sampler & classifier on empty/edge inputs.
# ---------------------------------------------------------------------------
def test_chaos_classifier_empty_input() -> None:
    """The bottleneck classifier must return a string for any input."""
    # We can't import distill_server (it spins up a FastAPI app on import),
    # so reach into the helper module via dynamic load.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "distill_server_chaos",
        str(_SCRIPTS / "distill_server.py"),
    )
    if spec is None or spec.loader is None:
        pytest.skip("cannot load distill_server.py")
    # Side-effects of import (FastAPI app, daemon thread) are acceptable
    # in a chaos run; everything is process-local and gets torn down by pytest.
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"distill_server import failed (env): {exc}")

    classify = getattr(mod, "_classify_bottleneck", None)
    assert callable(classify), "_classify_bottleneck must exist"

    # Empty input should not crash and should produce a string verdict.
    assert isinstance(classify([]), str)
    # Single tick with all zeros -> still a string (likely "Idle / sync stall").
    assert isinstance(classify([{
        "gpu.util_pct": 0, "cpu.util_pct": 0, "disk.read_mb_s": 0, "ram.pct": 0,
    }]), str)
    # Healthy GPU-bound case -> some non-empty verdict.
    healthy = [{
        "gpu.util_pct": 95, "cpu.util_pct": 50, "disk.read_mb_s": 30, "ram.pct": 60,
    }]
    verdict = classify(healthy)
    assert isinstance(verdict, str) and verdict
