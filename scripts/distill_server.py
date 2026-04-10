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

"""University of Distillation -- FastAPI server.

Replaces the Gradio distill_ui.py with a proper FastAPI + SSE backend.
The frontend is distill.html (served as a static file).

Launch:
    python scripts/distill_server.py [--port 7870]
"""

from __future__ import annotations

import asyncio
import glob
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import uvicorn  # xray: ignore[SEC-015]
from fastapi import FastAPI  # xray: ignore[SEC-015]
from fastapi.middleware.cors import CORSMiddleware  # xray: ignore[SEC-015]
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse  # xray: ignore[SEC-015]
from pydantic import BaseModel  # xray: ignore[SEC-015]
from zen_core_libs.llm import (  # xray: ignore[SEC-015]
    background_fill_gguf_cache as _zcl_bg_cache,
)

# ---------------------------------------------------------------------------
# zen_core_libs — shared GGUF utilities (model scan, metadata, memory est.)
# ---------------------------------------------------------------------------
from zen_core_libs.llm import (  # xray: ignore[SEC-015]
    discover_models as _zcl_discover_models,
)
from zen_core_libs.llm import (  # xray: ignore[SEC-015]
    estimate_model_memory_gb as _zcl_estimate_ram,
)
from zen_core_libs.llm import (  # xray: ignore[SEC-015]
    infer_metadata_from_filename as _zcl_infer_meta,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parent
PYTHON = sys.executable
FRONTEND_HTML = SCRIPTS_DIR / "distill.html"

sys.path.insert(0, str(SCRIPTS_DIR))

# ---------------------------------------------------------------------------
# Process guard — kills child process trees on exit/signal so a crashed
# server doesn't leave a 56 GB multi_teacher_generate.py orphaned in RAM.
# Imported eagerly so install() runs before any subprocess is spawned.
# ---------------------------------------------------------------------------
try:
    import process_guard  # type: ignore  # xray: ignore[SEC-015]
    _PROCESS_GUARD_OK = True
except ImportError:
    process_guard = None  # type: ignore[assignment]
    _PROCESS_GUARD_OK = False

# ---------------------------------------------------------------------------
# Rich model metadata parser
# ---------------------------------------------------------------------------

_ARCH_DB: list[tuple[str, str, str, str]] = [
    # (keyword_in_lower, display_name, hex_color, maker)
    ("devstral",   "Devstral",  "#e11d48", "Mistral AI"),
    ("deepseek",   "DeepSeek",  "#06b6d4", "DeepSeek"),
    ("gemma",      "Gemma",     "#4285f4", "Google"),
    ("qwen",       "Qwen",      "#ff6900", "Alibaba"),
    ("mistral",    "Mistral",   "#7c3aed", "Mistral AI"),
    ("llama",      "LLaMA",     "#1877f2", "Meta"),
    ("phi",        "Phi",       "#0078d4", "Microsoft"),
    ("bitnet",     "BitNet",    "#f59e0b", "Microsoft"),
    ("glm",        "GLM",       "#22c55e", "THUDM"),
    ("smollm",     "SmolLM",    "#8b5cf6", "HuggingFace"),
    ("tinyllama",  "TinyLlama", "#64748b", "Community"),
    ("lfm",        "LFM",       "#f97316", "Liquid AI"),
    ("nerdsking",  "NerdsKing", "#84cc16", "Community"),
    ("gpt",        "GPT",       "#10a37f", "OpenAI/MS"),
]

_QUANT_DB: list[tuple[str, str, int]] = [
    # (pattern_lower, label, quality_pct)
    ("f32",    "F32",   100),
    ("f16",    "F16",    98),
    ("q8_0",   "Q8",     95),
    ("q6_k",   "Q6K",    87),
    ("q5_k_l", "Q5KL",   84),
    ("q5_k_m", "Q5KM",   82),
    ("q5_k_s", "Q5KS",   79),
    ("q4_k_m", "Q4KM",   72),
    ("q4_k_s", "Q4KS",   69),
    ("q4_0",   "Q4",     66),
    ("q3_k_m", "Q3KM",   57),
    ("q2_k",   "Q2K",    46),
    ("i2_s",   "I2S",    43),
    ("mxfp4",  "MXFP4",  70),
]


def _parse_model_meta(name: str, path: str, size_gb: float) -> dict:
    """Extract rich display metadata from a GGUF filename.

    Uses zen_core_libs for base metadata (quant, arch, params) and memory
    estimation; layers on UI-specific colours, roles, and capabilities.
    """
    n = name.lower()

    is_proj = "mmproj" in n or n.startswith("proj")

    # ── Base metadata via zen_core_libs ───────────────────────────────────
    zcl = _zcl_infer_meta(path)
    zcl_arch = zcl.get("architecture", "")      # lowercase, e.g. "qwen"
    zcl_quant = zcl.get("quantization", "")      # canonical, e.g. "Q4_K_M"
    params = zcl.get("parameters", "")            # e.g. "14B"

    # Architecture display lookup (UI-specific colours / makers)
    arch, color, maker = "LLM", "#718096", "Unknown"
    for key, disp, col, mkr in _ARCH_DB:
        if key in n or key == zcl_arch:
            arch, color, maker = disp, col, mkr
            break

    # Quantization display label + quality score (UI-specific)
    quant, quant_quality = zcl_quant or "?", 70
    for pat, lbl, q in _QUANT_DB:
        if pat in n:
            quant, quant_quality = lbl, q
            break

    # Capabilities inferred from name
    caps: list[str] = []
    rules = [
        (["coder", "code", "devstral", "nerdsking", "python"], "coding"),
        (["r1", "distill-qwen", "reason", "think"],             "reasoning"),
        (["math"],                                               "math"),
        (["flash", "mini", "tiny", "small", "smol", "0.5b", "1.1b", "1.2b", "135m"], "fast"),
        (["gemma-4", "gemma4", "vl", "vlm", "vision"],          "multimodal"),
        (["tool", "agentic", "devstral", "function"],            "tools"),
        (["glm", "-zh-", "chinese"],                             "chinese"),
        (["instruct", "chat", "it-q", "-it.", "sft"],            "instruction"),
        (["safety", "safe"],                                     "safety"),
    ]
    for keywords, cap in rules:
        if any(k in n for k in keywords):
            caps.append(cap)
    if arch in ("Qwen", "GLM", "Gemma") and "chinese" not in caps:
        caps.append("multilingual")
    if not any(c in caps for c in ("coding", "math", "multimodal", "chinese", "reasoning")):
        caps.append("general")
    # dedupe
    seen: set[str] = set()
    caps = [c for c in caps if not (c in seen or seen.add(c))]  # type: ignore[func-returns-value]

    # Role hint
    if is_proj:
        role, role_label, role_color = "projection", "Projection", "#718096"
    elif size_gb >= 14:
        role, role_label, role_color = "dean", "Dean", "#f59e0b"
    elif size_gb >= 3.5:
        role, role_label, role_color = "teacher", "Teacher", "#3b82f6"
    else:
        role, role_label, role_color = "student", "Student", "#10b981"

    # RAM estimate — zen_core_libs accounts for quant-specific overhead
    ram_gb = _zcl_estimate_ram(int(size_gb * 1024), zcl_quant)

    # Clean display name: strip quant suffix, vendor prefix
    display = Path(path).stem
    for pat, _, _ in _QUANT_DB:
        display = re.sub(r"[-_.]" + re.escape(pat) + r"$", "", display, flags=re.IGNORECASE)
    display = re.sub(r"^(google_|ms_|msft_)", "", display, flags=re.IGNORECASE)
    display = display.rstrip("-_.")
    # Append quant tag so different quantizations are distinguishable
    if quant:
        display = f"{display} ({quant})"

    # Human-readable quant explanation
    _QUANT_TIPS: dict[str, str] = {
        "F32":   "Full 32-bit -- lossless, huge RAM cost",
        "F16":   "Half 16-bit -- near-lossless, 2x smaller than F32",
        "Q8":    "8-bit -- excellent quality, ~2x smaller than F16",
        "Q6K":   "6-bit -- very good quality, noticeable RAM savings",
        "Q5KL":  "5-bit large -- good quality, solid RAM/quality balance",
        "Q5KM":  "5-bit medium -- good quality, recommended sweet-spot",
        "Q5KS":  "5-bit small -- decent, slightly lower than Q5KM",
        "Q4KM":  "4-bit medium -- popular choice, 4x smaller than F16, minor quality loss",
        "Q4KS":  "4-bit small -- compact, some quality loss on hard tasks",
        "Q4":    "4-bit basic -- smaller than Q4KM but less accurate",
        "Q3KM":  "3-bit -- significant quality loss, use only if RAM-limited",
        "Q2K":   "2-bit -- heavy quality loss, last resort for tiny RAM",
        "I2S":   "2-bit importance -- experimental, lowest quality",
        "MXFP4": "Microscaling FP4 -- hardware-accelerated 4-bit",
    }
    quant_tip = _QUANT_TIPS.get(quant, f"{quant} quantization")

    return {
        "path":          path,
        "name":          Path(path).stem,
        "display_name":  display,
        "size_gb":       size_gb,
        "ram_gb":        ram_gb,
        "arch":          arch,
        "arch_color":    color,
        "maker":         maker,
        "params":        params,
        "quant":         quant,
        "quant_quality": quant_quality,
        "quant_tip":     quant_tip,
        "caps":          caps,
        "role":          role,
        "role_label":    role_label,
        "role_color":    role_color,
        "is_projection": is_proj,
    }

# ---------------------------------------------------------------------------
# Suppress noisy 404 spam from stale Gradio browser tabs
# ---------------------------------------------------------------------------
class _NoGradioFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "gradio_api" not in (record.getMessage().lower())


logging.getLogger("uvicorn.access").addFilter(_NoGradioFilter())

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="University of Distillation", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Iter 1 metrics rail -- per-second telemetry sampler
# ---------------------------------------------------------------------------
# Samples six categories every second (GPU/CPU/RAM/Disk/Net/Pipeline) and
# pushes them to:
#   1. an in-memory ring buffer (last 60 ticks) accessible via /api/metrics
#   2. an append-only metrics.jsonl sidecar in the active output_dir
#   3. the existing /api/pipeline/start SSE stream as `type: "metrics"` frames
#
# psutil is a hard requirement; nvidia-smi is best-effort (returns zeros on
# CPU-only boxes). The sampler runs in a daemon thread started at module import.

try:
    import psutil  # xray: ignore[SEC-015]
    _PSUTIL_OK = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    _PSUTIL_OK = False

_METRICS_RING: list[dict] = []
_METRICS_RING_MAX = 60
_METRICS_LOCK = threading.Lock()
_METRICS_SINK_PATH: Path | None = None  # set by /api/pipeline/start
_METRICS_PIPELINE_STAGE: str = "idle"
_METRICS_TOK_S: float = 0.0
_METRICS_ETA: str = ""

# Pipeline cancel machinery -- the Studio UI is the ONE place the pipeline
# can be launched, but any tab (or /api/pipeline/stop) can cancel it. Only
# one pipeline runs at a time, so a single active-proc slot is enough.
_PIPELINE_LOCK = threading.Lock()
_ACTIVE_PROC: subprocess.Popen | None = None
_CANCEL_REQUESTED: bool = False


def _read_nvidia_smi() -> dict:
    """Best-effort GPU stats via nvidia-smi. Returns zeros on failure."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return {
            "gpu.util_pct": 0.0,
            "gpu.mem_used_mb": 0.0,
            "gpu.mem_total_mb": 0.0,
            "gpu.power_w": 0.0,
            "gpu.temp_c": 0.0,
        }
    line = out.decode("utf-8", errors="ignore").strip().splitlines()
    if not line:
        return {
            "gpu.util_pct": 0.0,
            "gpu.mem_used_mb": 0.0,
            "gpu.mem_total_mb": 0.0,
            "gpu.power_w": 0.0,
            "gpu.temp_c": 0.0,
        }
    # If multiple GPUs, average the first 4 fields and sum memory.
    util_vals: list[float] = []
    mem_used: list[float] = []
    mem_total: list[float] = []
    power_vals: list[float] = []
    temp_vals: list[float] = []
    for row in line:
        parts = [p.strip() for p in row.split(",")]
        if len(parts) < 5:
            continue
        try:
            util_vals.append(float(parts[0]))
            mem_used.append(float(parts[1]))
            mem_total.append(float(parts[2]))
            power_vals.append(float(parts[3]))
            temp_vals.append(float(parts[4]))
        except ValueError:
            continue
    if not util_vals:
        return {
            "gpu.util_pct": 0.0,
            "gpu.mem_used_mb": 0.0,
            "gpu.mem_total_mb": 0.0,
            "gpu.power_w": 0.0,
            "gpu.temp_c": 0.0,
        }
    return {
        "gpu.util_pct":   sum(util_vals) / len(util_vals),
        "gpu.mem_used_mb": sum(mem_used),
        "gpu.mem_total_mb": sum(mem_total),
        "gpu.power_w":    sum(power_vals) / len(power_vals),
        "gpu.temp_c":     max(temp_vals),
    }


_LAST_DISK_IO: dict | None = None
_LAST_NET_IO: dict | None = None
_LAST_IO_TS: float = 0.0


def _sample_metrics_once() -> dict:
    """Sample one metric tick from psutil + nvidia-smi. Cheap (~1 ms)."""
    global _LAST_DISK_IO, _LAST_NET_IO, _LAST_IO_TS  # noqa: PLW0603
    now = time.time()
    tick: dict = {
        "ts": now,
        "stage": _METRICS_PIPELINE_STAGE,
        "tok_s": _METRICS_TOK_S,
        "eta": _METRICS_ETA,
    }
    if _PSUTIL_OK:
        try:
            tick["cpu.util_pct"] = float(psutil.cpu_percent(interval=None))
            vm = psutil.virtual_memory()
            tick["ram.used_gb"] = vm.used / 1e9
            tick["ram.total_gb"] = vm.total / 1e9
            tick["ram.pct"] = vm.percent
            sw = psutil.swap_memory()
            tick["swap.used_gb"] = sw.used / 1e9
            disk = psutil.disk_usage("/")
            tick["disk.free_gb"] = disk.free / 1e9
            io = psutil.disk_io_counters()
            net = psutil.net_io_counters()
            elapsed = max(now - _LAST_IO_TS, 0.001) if _LAST_IO_TS else 1.0
            if io is not None:
                if _LAST_DISK_IO is not None:
                    tick["disk.read_mb_s"]  = max(0.0, (io.read_bytes  - _LAST_DISK_IO["read_bytes"])  / elapsed / 1e6)
                    tick["disk.write_mb_s"] = max(0.0, (io.write_bytes - _LAST_DISK_IO["write_bytes"]) / elapsed / 1e6)
                else:
                    tick["disk.read_mb_s"] = 0.0
                    tick["disk.write_mb_s"] = 0.0
                _LAST_DISK_IO = {"read_bytes": io.read_bytes, "write_bytes": io.write_bytes}
            if net is not None:
                if _LAST_NET_IO is not None:
                    tick["net.tx_mb_s"] = max(0.0, (net.bytes_sent - _LAST_NET_IO["bytes_sent"]) / elapsed / 1e6)
                    tick["net.rx_mb_s"] = max(0.0, (net.bytes_recv - _LAST_NET_IO["bytes_recv"]) / elapsed / 1e6)
                else:
                    tick["net.tx_mb_s"] = 0.0
                    tick["net.rx_mb_s"] = 0.0
                _LAST_NET_IO = {"bytes_sent": net.bytes_sent, "bytes_recv": net.bytes_recv}
            _LAST_IO_TS = now
        except (OSError, AttributeError):
            pass
    tick.update(_read_nvidia_smi())
    return tick


def _classify_bottleneck(ticks: list[dict]) -> str:
    """One-line bottleneck verdict over the most recent ticks."""
    if not ticks:
        return "idle"
    n = len(ticks)
    avg_gpu = sum(float(t.get("gpu.util_pct", 0) or 0) for t in ticks) / n
    avg_cpu = sum(float(t.get("cpu.util_pct", 0) or 0) for t in ticks) / n
    avg_io_r = sum(float(t.get("disk.read_mb_s", 0) or 0) for t in ticks) / n
    ram_pct = float(ticks[-1].get("ram.pct", 0) or 0)
    if avg_gpu < 50 and avg_cpu > 80:
        return "CPU-bound (dataloader?)"
    if avg_gpu < 50 and avg_io_r > 200:
        return "I/O-bound (slow disk?)"
    if avg_gpu < 50 and avg_cpu < 50:
        return "Idle / sync stall"
    if avg_gpu > 90 and ram_pct > 92:
        return "VRAM/RAM-pressured"
    if avg_gpu > 80:
        return "GPU-bound (healthy)"
    return "Mixed"


def _metrics_loop() -> None:
    """Background sampler -- 1 Hz. Started as a daemon at import time."""
    while True:
        try:
            tick = _sample_metrics_once()
            with _METRICS_LOCK:
                _METRICS_RING.append(tick)
                if len(_METRICS_RING) > _METRICS_RING_MAX:
                    _METRICS_RING.pop(0)
                sink = _METRICS_SINK_PATH
            if sink is not None:
                try:
                    with sink.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(tick) + "\n")
                except OSError:
                    pass
        except Exception:  # noqa: BLE001 - sampler must never crash
            pass
        time.sleep(1.0)


# Start the sampler thread once, at import time. Daemon = dies with the server.
threading.Thread(target=_metrics_loop, daemon=True, name="metrics-sampler").start()


def _set_metrics_sink(output_dir: Path | None) -> None:
    """Point the sampler at a metrics.jsonl sidecar in the run output_dir."""
    global _METRICS_SINK_PATH  # noqa: PLW0603
    if output_dir is None:
        _METRICS_SINK_PATH = None
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    _METRICS_SINK_PATH = output_dir / "metrics.jsonl"


def _set_metrics_stage(stage: str) -> None:
    global _METRICS_PIPELINE_STAGE  # noqa: PLW0603
    _METRICS_PIPELINE_STAGE = stage


def _set_metrics_throughput(tok_s: float, eta: str = "") -> None:
    global _METRICS_TOK_S, _METRICS_ETA  # noqa: PLW0603
    _METRICS_TOK_S = tok_s
    _METRICS_ETA = eta


@app.get("/api/metrics")
async def get_metrics(history: int = 30) -> JSONResponse:
    """Return the most recent metric ticks + bottleneck classification.

    Query params:
        history: number of recent ticks to return (default 30, max 60).
    """
    history = max(1, min(int(history or 30), _METRICS_RING_MAX))
    with _METRICS_LOCK:
        ticks = list(_METRICS_RING[-history:])
    return JSONResponse({
        "current": ticks[-1] if ticks else {},
        "history": ticks,
        "bottleneck": _classify_bottleneck(ticks),
        "psutil_ok": _PSUTIL_OK,
    })


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

# Regex for tqdm progress lines from HuggingFace Trainer:
#   "  5%|...| 15/300 [00:45<14:15, 3.00s/it]"
_TQDM_RE = re.compile(
    r"(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[([^\]<]+)<([^\],]+)"
)

# Regex for LlamaFactory LogCallback JSON-in-stdout:
#   "{'loss': 2.1000, 'learning_rate': 2.0e-04, 'epoch': 0.50, 'total_steps': 300}"
_LOGCB_RE = re.compile(
    r"'loss':\s*([\d.]+).*?'epoch':\s*([\d.]+)"
)


def _parse_train_progress(line: str) -> dict | None:
    """Try to extract training progress from a log line.

    Returns a dict with pct, step, total, eta, loss, epoch (all optional)
    or None if the line doesn't contain progress info.
    """
    # Try tqdm bar first (most reliable)
    m = _TQDM_RE.search(line)
    if m:
        pct, step, total, elapsed, eta = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        return {
            "pct": int(pct), "step": int(step), "total": int(total),
            "elapsed": elapsed.strip(), "eta": eta.strip(),
        }

    # Try reading trainer_log.jsonl JSON lines (written by LogCallback)
    stripped = line.strip()
    if stripped.startswith("{") and "current_steps" in stripped and "total_steps" in stripped:
        try:
            d = json.loads(stripped)
            return {
                "pct": d.get("percentage", 0),
                "step": d.get("current_steps", 0),
                "total": d.get("total_steps", 0),
                "elapsed": d.get("elapsed_time", ""),
                "eta": d.get("remaining_time", ""),
                "loss": d.get("loss"),
                "lr": d.get("lr"),
                "epoch": d.get("epoch"),
            }
        except (json.JSONDecodeError, KeyError):
            pass

    # Try LogCallback stdout format: {'loss': 2.1000, ...}
    m2 = _LOGCB_RE.search(line)
    if m2:
        return {"loss": float(m2.group(1)), "epoch": float(m2.group(2))}

    return None


def _run_script(cmd: list[str], timeout: int = 7200) -> tuple[int, str, str]:
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, cwd=str(ROOT_DIR),
    )
    return result.returncode, result.stdout or "", result.stderr or ""


def _popen(cmd: list[str]) -> subprocess.Popen:
    """Spawn a child process AND register it as the active pipeline proc.

    The UI exposes a STOP button that hits /api/pipeline/stop; that endpoint
    kills whatever proc is currently registered here, which lets a user
    cancel the pipeline mid-stage without needing to close the browser tab.

    The PID is also registered with ``process_guard`` so that an unclean
    server exit (Ctrl-C, signal, crash) walks the entire descendant tree
    and force-kills it instead of leaving orphans like the 56 GB
    multi_teacher_generate.py we observed in the wild.
    """
    global _ACTIVE_PROC  # noqa: PLW0603
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(ROOT_DIR),
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    with _PIPELINE_LOCK:
        _ACTIVE_PROC = proc
    if _PROCESS_GUARD_OK and process_guard is not None:
        # Use the script name as a human-friendly label so the recovery
        # file is readable
        label = " ".join(cmd[:2]) if len(cmd) >= 2 else (cmd[0] if cmd else f"pid={proc.pid}")
        process_guard.register_child(proc.pid, label)
    return proc


def _clear_active_proc(proc: subprocess.Popen | None = None) -> None:
    """Unregister the active proc once a stage finishes cleanly.

    If ``proc`` is passed, we only clear the slot when it still points at
    that specific proc (avoids a race where a later stage has already
    registered its own subprocess).

    Also unregisters from ``process_guard`` so the recovery file stays
    accurate.
    """
    global _ACTIVE_PROC  # noqa: PLW0603
    with _PIPELINE_LOCK:
        if proc is None or _ACTIVE_PROC is proc:
            _ACTIVE_PROC = None
    if proc is not None and _PROCESS_GUARD_OK and process_guard is not None:
        process_guard.unregister_child(proc.pid)


def _is_cancel_requested() -> bool:
    with _PIPELINE_LOCK:
        return _CANCEL_REQUESTED


def _reset_cancel_flag() -> None:
    global _CANCEL_REQUESTED  # noqa: PLW0603
    with _PIPELINE_LOCK:
        _CANCEL_REQUESTED = False


def _request_cancel() -> dict:
    """Flip the cancel flag and kill the active subprocess + its descendants.

    proc.kill() alone only kills the direct child. Many of our pipeline
    stages (multi_teacher_generate, llamafactory-cli train) spawn their own
    subprocesses (vllm, llama-server, torch.distributed workers) which
    would be orphaned by a shallow kill — that's the bug that left a 56 GB
    multi_teacher_generate.py running after a crash. Using
    ``process_guard.kill_all()`` walks the registered tree and force-kills
    every descendant, which is what the STOP button should have done all
    along.
    """
    global _CANCEL_REQUESTED  # noqa: PLW0603
    with _PIPELINE_LOCK:
        _CANCEL_REQUESTED = True
        proc = _ACTIVE_PROC
    killed = False
    tree_killed = 0
    if _PROCESS_GUARD_OK and process_guard is not None:
        tree_killed = process_guard.kill_all()
        killed = tree_killed > 0
    elif proc is not None and proc.poll() is None:
        # Fallback: shallow kill if process_guard isn't available
        try:
            proc.kill()
            killed = True
        except OSError:
            pass
    return {
        "cancel_requested": True,
        "killed_proc": killed,
        "had_active_proc": proc is not None,
        "tree_killed": tree_killed,
    }


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------

def _sse(data: Any) -> str:
    """Encode one SSE data frame."""
    return f"data: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# Single-file prompt count
# ---------------------------------------------------------------------------

def _safe_prompt_count(path: str) -> int:
    p = Path(path)
    if not p.is_file():
        return 0
    try:
        raw = p.read_text(encoding="utf-8").strip()
        if not raw:
            return 0
        if raw.startswith("["):
            rows = json.loads(raw)
            return len(rows) if isinstance(rows, list) else 0
        return sum(1 for l in raw.splitlines() if l.strip())
    except (OSError, json.JSONDecodeError, ValueError):
        return 0


def _sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


# ---------------------------------------------------------------------------
# Zena Dean singleton (avoids mutable-global pattern)
# ---------------------------------------------------------------------------
class _DeanHolder:
    """Lazy singleton for the Zena Dean instance."""

    __slots__ = ("_instance", "_gguf")

    def __init__(self) -> None:
        self._instance: Any = None
        self._gguf: str = ""

    def get(self, gguf: str) -> Any:
        if self._instance is None or self._gguf != gguf:
            from zena_dean import ZenaDean
            self._instance = ZenaDean(gguf)
            self._gguf = gguf
        return self._instance


_dean_holder = _DeanHolder()


def _get_dean(gguf: str):
    return _dean_holder.get(gguf)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    return FileResponse(str(FRONTEND_HTML))


@app.get("/zena.png")
async def zena_avatar():
    """Serve the Zena avatar image used in the top bar hamburger/settings button."""
    # Prefer the 256x256 version for crispness on hi-DPI displays; fall back to zena.png.
    for candidate in (ROOT_DIR / "zena_256x256.png", ROOT_DIR / "zena.png"):
        if candidate.exists():
            return FileResponse(str(candidate), media_type="image/png")
    return JSONResponse({"error": "zena.png not found"}, status_code=404)


# ── Scan for GGUF models ────────────────────────────────────────────────────

@app.get("/api/scan")
async def scan_models():
    # ── Collect GGUF paths from zen_core_libs + extra local dirs ──────────
    seen: set[str] = set()
    raw_models: list[dict] = []

    # zen_core_libs scans $MODELS_DIR, ~/AI/Models, and its bundled models/
    for m in _zcl_discover_models():
        p = os.path.normpath(m["path"])
        if p not in seen:
            seen.add(p)
            raw_models.append({"path": p, "name": m["name"], "size_gb": m["size_gb"]})

    # Extra dirs the server knows about (may overlap — deduped via seen)
    extra_dirs = [
        r"C:\AI\Models",
        r"C:\AI",
        os.path.expanduser("~/.cache/llama.cpp"),
    ]
    for d in extra_dirs:
        pattern = os.path.join(d, "**", "*.gguf")
        for path in glob.glob(pattern, recursive=True):
            path = os.path.normpath(path)
            if path in seen:
                continue
            seen.add(path)
            try:
                size_gb = round(os.path.getsize(path) / 1024**3, 1)
            except OSError:
                size_gb = 0.0
            raw_models.append({"path": path, "name": Path(path).name, "size_gb": size_gb})

    # ── Enrich with UI metadata ──────────────────────────────────────────
    found: list[dict] = []
    for m in raw_models:
        meta = _parse_model_meta(Path(m["path"]).stem, m["path"], m["size_gb"])
        found.append(meta)

    # Sort: dean > teacher > student > projection; alpha within group
    order = {"dean": 0, "teacher": 1, "student": 2, "projection": 3}
    found.sort(key=lambda m: (order.get(m["role"], 9), m["display_name"].lower()))

    # ── Background GGUF header cache warming (zen_core_libs disk cache) ──
    all_paths = [m["path"] for m in found]
    threading.Thread(target=_zcl_bg_cache, args=(all_paths,), daemon=True).start()

    return JSONResponse(found)


# ── Credential check (fast name-heuristic, optional bg deep probe) ──────────

class CredCheckReq(BaseModel):
    teachers: list[str]
    backend: str = "inprocess"
    output_dir: str = "data/purified"
    deep_probe: bool = False
    zena_gguf: str = ""


@app.post("/api/credential-check")
async def credential_check(req: CredCheckReq):
    if not req.teachers:
        return JSONResponse({"error": "No teachers provided."}, status_code=400)

    out = Path(req.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    profile_path = str(out / "teacher_profile.json")

    # Fast in-process name heuristic — always instant
    try:
        from teacher_profiler import profile_all_teachers
        profile_obj = profile_all_teachers(req.teachers, query_fn=None, probe_caps=None)
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile_obj, f, indent=2, ensure_ascii=False)
    except (ImportError, OSError, ValueError) as exc:
        return JSONResponse({"error": f"Name-heuristic failed: {exc}"}, status_code=500)

    probe_msg: str | None = None
    if req.deep_probe:
        # Fire-and-forget background process — never block the response
        cmd = [
            PYTHON, str(SCRIPTS_DIR / "teacher_profiler.py"),
            "--teachers", *req.teachers,
            "--backend", req.backend,
            "--output", profile_path,
            "--probe",
        ]
        log_path = out / "deep_probe.log"
        try:
            with open(log_path, "w", encoding="utf-8") as lg:
                subprocess.Popen(cmd, stdout=lg, stderr=subprocess.STDOUT,
                                 cwd=str(ROOT_DIR), close_fds=True)
            probe_msg = f"Deep probe running in background. Log: {log_path}"
        except OSError as exc:
            probe_msg = f"Could not start background probe: {exc}"

    # Load and return profile
    try:
        with open(profile_path, encoding="utf-8") as f:
            profile = json.load(f)
    except (OSError, json.JSONDecodeError):
        profile = {}

    # Enroll with Zena (best-effort)
    if req.zena_gguf.strip():
        try:
            dean = _get_dean(req.zena_gguf)
            for t in req.teachers:
                dean.enroll_teacher(t)
        except (OSError, ValueError, RuntimeError):
            pass

    return JSONResponse({
        "profile": profile,
        "profile_path": profile_path,
        "probe_msg": probe_msg,
    })


# ── Phase 2 progress (fast, reads checkpoint files) ─────────────────────────

@app.get("/api/progress")
async def phase2_progress(
    teachers: str,
    prompts_path: str,
    output_dir: str = "data/purified",
):
    teacher_list = [t.strip() for t in teachers.split("\n") if t.strip()]
    prompt_count = _safe_prompt_count(prompts_path)
    ckpt_dir = Path(output_dir) / "checkpoints"

    rows: list[dict] = []
    total_done = 0
    elapsed_samples: list[float] = []

    for tp in teacher_list:
        stem = Path(tp).stem
        ckpt = ckpt_dir / f"{_sanitize(stem)}.jsonl"
        done = 0
        last_update = None
        if ckpt.is_file():
            with open(ckpt, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    done += 1
                    try:
                        rec = json.loads(line)
                        s = rec.get("response", {}).get("elapsed_s")
                        if isinstance(s, (int, float)) and s > 0:
                            elapsed_samples.append(float(s))
                    except (json.JSONDecodeError, KeyError, ValueError):
                        pass
            try:
                last_update = int(time.time() - ckpt.stat().st_mtime)
            except OSError:
                pass
        done = min(done, prompt_count) if prompt_count else done
        pct = round(100.0 * done / prompt_count, 1) if prompt_count else 0
        total_done += done
        rows.append({"teacher": stem, "done": done, "total": prompt_count, "pct": pct, "age_s": last_update})

    total_target = prompt_count * len(teacher_list) if prompt_count else 0
    eta_s: float | None = None
    if elapsed_samples and total_target > total_done:
        trimmed = sorted(elapsed_samples)[:max(1, int(len(elapsed_samples) * 0.9))]
        avg_s = sum(trimmed) / len(trimmed)
        eta_s = round((total_target - total_done) * avg_s)

    return JSONResponse({
        "rows": rows,
        "total_done": total_done,
        "total_target": total_target,
        "pct": round(100.0 * total_done / total_target, 1) if total_target else 0,
        "eta_s": eta_s,
    })


# ── Zena chat ────────────────────────────────────────────────────────────────

class ChatReq(BaseModel):
    message: str
    history: list[dict] = []
    zena_gguf: str = ""
    teachers_text: str = ""


@app.post("/api/chat")
async def chat(req: ChatReq):
    msg = req.message.strip()
    msg_lower = msg.lower()

    def _reply(text: str) -> JSONResponse:
        return JSONResponse({"reply": text})

    # scan
    if msg_lower in ("scan", "discover", "find models", "list models"):
        resp = await scan_models()
        try:
            models = json.loads(resp.body)
        except (json.JSONDecodeError, TypeError):
            return _reply("Scan returned invalid data.")
        if not models:
            return _reply("No .gguf models found in default directories.")
        lines = [f"Found **{len(models)}** GGUF model(s):\n"]
        for m in models[:30]:
            lines.append(f"- `{m['path']}` ({m['size_gb']} GB)")
        return _reply("\n".join(lines))

    # status
    if msg_lower in ("status", "who", "enrolled", "professors"):
        if not req.zena_gguf.strip():
            return _reply("Set the Zena GGUF path first.")
        dean = _get_dean(req.zena_gguf)
        return _reply(dean.get_enrollment_summary())

    # cross-examine
    if msg_lower in ("cross-examine", "cross examine", "exam", "peer review"):
        if not req.zena_gguf.strip():
            return _reply("Set the Zena GGUF path first.")
        dean = _get_dean(req.zena_gguf)
        if len(dean.teachers) < 2:
            return _reply("Need at least **2 enrolled professors** first.")
        dean.cross_examine()
        return _reply(dean.get_peer_matrix_markdown())

    # recommend
    if msg_lower.startswith("recommend"):
        goal = msg[len("recommend"):].strip() or "general distillation"
        if not req.zena_gguf.strip():
            return _reply("Set the Zena GGUF path first.")
        dean = _get_dean(req.zena_gguf)
        return _reply(dean.recommend_teachers(goal))

    # enroll
    if msg_lower.startswith(("enroll", "add")):
        if not req.zena_gguf.strip():
            return _reply("Set the Zena GGUF path first.")
        dean = _get_dean(req.zena_gguf)
        paths = [l.strip() for l in req.teachers_text.splitlines() if l.strip()]
        if not paths:
            return _reply("Paste teacher GGUF paths in the enrollment box first.")
        results = []
        for p in paths:
            info = dean.enroll_teacher(p)
            caps = ", ".join(info.get("top_capabilities", [])[:3])
            results.append(f"- **{Path(p).stem}** -- {caps}")
        return _reply("Enrolled:\n" + "\n".join(results) + "\n\n" + dean.get_enrollment_summary())

    # LLM fallback
    if not req.zena_gguf.strip():
        return _reply(
            "Hello! I'm **Zena**, Dean of this University.\n\n"
            "Set my GGUF path, then try: `scan`, `enroll`, `cross-examine`, `status`, `recommend <goal>`."
        )
    try:
        dean = _get_dean(req.zena_gguf)
        return _reply(dean.chat(msg))
    except Exception as exc:
        return _reply(f"*(LLM error: {exc})*")


# ── Cross-examination ────────────────────────────────────────────────────────

class CrossExamReq(BaseModel):
    zena_gguf: str


@app.post("/api/cross-exam")
async def cross_exam(req: CrossExamReq):
    if not req.zena_gguf.strip():
        return JSONResponse({"error": "No Zena GGUF path."}, status_code=400)
    try:
        dean = _get_dean(req.zena_gguf)
        if len(getattr(dean, "teachers", [])) < 2:
            return JSONResponse({"error": "Need at least 2 enrolled teachers."}, status_code=400)
        dean.cross_examine()
        return JSONResponse({"matrix_md": dean.get_peer_matrix_markdown()})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


# ── Generate training configs ────────────────────────────────────────────────

class GenConfigReq(BaseModel):
    output_dir: str = "data/purified"
    student_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    cpu_safe: bool = True
    tag: str = "distill_auto"


@app.post("/api/gen-configs")
async def gen_configs(req: GenConfigReq):
    config_out = str(ROOT_DIR / "examples" / "distillation" / "auto")
    cmd = [
        PYTHON, str(SCRIPTS_DIR / "gen_distill_configs.py"),
        "--student", req.student_model,
        "--data-dir", req.output_dir,
        "--out-dir", config_out,
        "--tag", req.tag,
    ]
    if req.cpu_safe:
        cmd.append("--cpu-safe")
    rc, stdout, stderr = _run_script(cmd, timeout=60)
    if rc != 0:
        return JSONResponse({"error": stderr[-2000:]}, status_code=500)
    yamls = sorted(glob.glob(os.path.join(config_out, "*.yaml")))
    return JSONResponse({"configs": yamls, "stdout": stdout})


# ── Full pipeline SSE stream ─────────────────────────────────────────────────

class PipelineReq(BaseModel):
    teachers: list[str]
    prompts_path: str
    backend: str = "inprocess"
    output_dir: str = "data/purified"
    max_tokens: int = 512
    temperature: float = 0.7
    answer_th: float = 0.6
    reason_th: float = 0.5
    enable_halluc: bool = True
    zena_gguf: str = ""
    profile_path: str = ""
    do_train: bool = True
    student_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    cpu_safe: bool = True
    saves_tag: str = "distill_auto"
    # Studio toggles -- honor the "Run eval after training" and
    # "Postmortem report on completion" checkboxes. Both default to True so
    # Recipes-style callers keep their historical behavior.
    run_eval: bool = True
    run_postmortem: bool = True
    # Curriculum: skills & languages the student enrolled for
    skills: list[str] = []       # e.g. ["translation", "ocr_cleanup"]
    languages: list[str] = []    # e.g. ["en", "de", "fr"]
    code_langs: list[str] = []   # e.g. ["python", "rust"]
    # Sequential batch training: when the Studio has 2+ enrolled students the
    # frontend submits the full list here. Stages 1-3 (teacher gen + halluc +
    # purify) still run ONCE shared, then stages 4-10 loop per student with a
    # per-student saves tag. If empty, behave exactly as before (single
    # student -- uses student_model / saves_tag / skills / languages from the
    # flat fields above).
    students_batch: list[dict] = []
    # ── Speed-Run mode: skip teacher generation entirely ──────────────────
    # When upstream_dataset is set we treat a pre-distilled HF reasoning
    # dataset (e.g. Roman1111111/claude-opus-4.6-10000x) as a pre-purified
    # GOLD set. Stages 1-3 (generate / halluc / purify) are short-circuited:
    # we just download the dataset, write consensus_sft.jsonl in seconds, and
    # jump straight to config generation + SFT. Goes from "I have nothing"
    # to "I'm training" in <60s for the small datasets. Set
    # upstream_max_rows to cap the import size for fast experiments.
    upstream_dataset: str = ""           # e.g. "Roman1111111/claude-opus-4.6-10000x"
    upstream_max_rows: int = 0           # 0 = unlimited


# Prompt-ID prefix → skill mapping
_PROMPT_SKILL_MAP: dict[str, str] = {
    "tr-": "translation",
    "chat-": "conversation",
    "ocr": "ocr_cleanup",
    "code-": "coding",
    "math-": "math",
    "reason-": "reasoning",
    "sum-": "summarize",
    "creative-": "creative",
}


def _filter_prompts_by_curriculum(
    src_path: str, dst_path: str,
    skills: list[str], languages: list[str],
) -> tuple[int, int]:
    """Filter a prompts JSONL file to keep only prompts matching the student's curriculum.

    Returns (original_count, filtered_count).
    If no skills specified, copies all prompts (no filtering).
    """
    src = Path(src_path)
    dst = Path(dst_path)
    if not src.is_file():
        return 0, 0

    kept: list[str] = []
    total = 0
    lang_set = {l.lower() for l in languages} if languages else set()

    for raw in open(src, encoding="utf-8"):
        raw = raw.strip()
        if not raw:
            continue
        total += 1
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        pid = row.get("id", "")

        # Determine the skill of this prompt from its ID prefix
        prompt_skill = ""
        for prefix, skill in _PROMPT_SKILL_MAP.items():
            if pid.startswith(prefix):
                prompt_skill = skill
                break

        # If we have skill filters, check membership
        if skills and prompt_skill and prompt_skill not in skills:
            continue

        # For translation prompts, filter by language pair
        if prompt_skill == "translation" and lang_set:
            # Extract language codes from ID like "tr-en2es-0013" or "tr-detect-0487"
            m = re.match(r"tr-(\w+)2(\w+)-", pid)
            if m:
                src_lang, tgt_lang = m.group(1).lower(), m.group(2).lower()
                if src_lang not in lang_set and tgt_lang not in lang_set:
                    continue
            # tr-detect prompts: keep if student has translation skill
            # (language detection is fundamental to translation)

        kept.append(raw)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(kept) + "\n" if kept else "", encoding="utf-8")
    return total, len(kept)


@app.post("/api/pipeline/start")
async def pipeline_start(req: PipelineReq):
    """Start the full pipeline and stream progress as SSE."""

    async def _gen() -> AsyncGenerator[str, None]:

        def _ev(stage: str, status: str, msg: str = "", **extra) -> str:
            # Side effect: keep the metrics rail in sync with the stage label.
            if status in ("running", "starting"):
                _set_metrics_stage(stage)
            return _sse({"type": "stage", "stage": stage, "status": status, "msg": msg, **extra})

        def _metrics_frame() -> str:
            """One SSE frame containing the latest metric tick + bottleneck verdict."""
            with _METRICS_LOCK:
                ticks = list(_METRICS_RING[-30:])
            if not ticks:
                return ""
            payload = {"type": "metrics", **ticks[-1], "bottleneck": _classify_bottleneck(ticks)}
            return _sse(payload)

        out = Path(req.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        out_jsonl = str(out / "teacher_responses.jsonl")
        tag = req.saves_tag.strip() or "distill_auto"
        saves_dir = ROOT_DIR / "saves" / tag
        saves_dir.mkdir(parents=True, exist_ok=True)
        config_out = ROOT_DIR / "examples" / "distillation" / "auto"

        # Iter 1: route the metrics sampler to write a sidecar jsonl into this run's
        # output directory so postmortem_agent.py can pick it up later.
        _set_metrics_sink(saves_dir)
        _set_metrics_stage("starting")
        # Fresh run -- clear any cancel flag left over from a previous run
        _reset_cancel_flag()

        # ── Speed-Run mode: import a pre-distilled HF dataset and skip
        # stages 1-3 entirely. Validation rules are different here -- no
        # teachers or prompts file required, just the upstream id.
        speed_run = bool(req.upstream_dataset.strip())
        if not speed_run:
            if not req.teachers:
                yield _sse({"type": "error", "msg": "No teachers listed."})
                return
            if not req.prompts_path or not Path(req.prompts_path).is_file():
                yield _sse({"type": "error", "msg": f"Prompts file not found: {req.prompts_path}"})
                return

        # ── Speed-Run mode (upstream HF dataset) ────────────────────────────
        # Short-circuit stages 1-3 by running scripts/import_reasoning_dataset.py
        # to download an already-distilled reasoning dataset (e.g. one of the
        # Claude 4.6 Opus public traces) and write it directly into
        # consensus_sft.jsonl. This goes from "click Start" to "training" in
        # ~30-60 seconds for small datasets, vs 30-300 minutes for the slow
        # path. The rest of the pipeline (configs -> SFT -> DPO -> merge ->
        # GGUF -> eval) runs unchanged.
        if speed_run:
            yield _ev("generate", "running",
                      f"Speed-Run: importing {req.upstream_dataset} ...")
            cmd_imp = [
                PYTHON, str(SCRIPTS_DIR / "import_reasoning_dataset.py"),
                "--dataset", req.upstream_dataset,
                "--output-dir", req.output_dir,
                "--tier", "GOLD",
            ]
            if req.upstream_max_rows and req.upstream_max_rows > 0:
                cmd_imp.extend(["--max-rows", str(int(req.upstream_max_rows))])
            proc_imp = _popen(cmd_imp)
            loop = asyncio.get_event_loop()
            assert proc_imp.stdout is not None
            while proc_imp.poll() is None:
                line = await loop.run_in_executor(None, proc_imp.stdout.readline)
                if line:
                    yield _sse({"type": "log", "stage": "generate", "text": line.rstrip()})
            rc_imp = proc_imp.wait()
            if rc_imp != 0:
                yield _ev("generate", "fail",
                          f"Speed-Run import failed (exit {rc_imp}).")
                yield _sse({"type": "error",
                            "msg": "Upstream dataset import failed -- see logs."})
                return

            # Read back the synthetic purification report so the rest of the
            # pipeline has the gold/silver/drop counts it expects.
            purify_rpt = out / "purification_report.json"
            try:
                with open(purify_rpt, encoding="utf-8") as f:
                    upstream_rpt = json.load(f)
                gold_n = int(upstream_rpt.get("gold_count", 0))
                silver_n = int(upstream_rpt.get("silver_count", 0))
                drop_n = int(upstream_rpt.get("dropped_count", 0))
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                yield _ev("generate", "fail",
                          f"Could not read purification_report.json: {exc}")
                return

            yield _ev("generate", "done",
                      f"Imported {gold_n} GOLD samples from {req.upstream_dataset}",
                      gold=gold_n, silver=silver_n, drop=drop_n)
            yield _ev("halluc", "skip", "Speed-Run: upstream dataset is pre-filtered.")
            yield _ev("purify", "skip",
                      f"Speed-Run: GOLD {gold_n} (no SILVER/DROP)",
                      gold=gold_n, silver=silver_n, drop=drop_n)

            # Persist a curriculum sidecar even in speed-run mode so dashboards
            # can report what trained on what.
            if req.skills:
                (saves_dir / "curriculum.json").write_text(
                    json.dumps({
                        "skills": req.skills,
                        "languages": req.languages,
                        "code_langs": req.code_langs,
                        "upstream_dataset": req.upstream_dataset,
                        "upstream_gold_count": gold_n,
                    }, indent=2),
                    encoding="utf-8",
                )

            # Set the same loop variables the slow-path stages would have set
            # so the rest of the function (do_train guard, GOLD=0 guard, the
            # per-student training loop) sees a consistent shape.
            n_prompts = gold_n
            n_teachers = 1
            actual_prompts = ""
            out_jsonl = str(out / "teacher_responses.jsonl")  # placeholder

        # ── Curriculum filter: keep only prompts matching enrolled skills ──
        if not speed_run:
            actual_prompts = req.prompts_path
        if req.skills and not speed_run:
            filtered_path = str(out / "curriculum_prompts.jsonl")
            orig_n, kept_n = _filter_prompts_by_curriculum(
                req.prompts_path, filtered_path, req.skills, req.languages,
            )
            if kept_n == 0:
                yield _sse({"type": "error",
                            "msg": f"Curriculum filter kept 0 of {orig_n} prompts. "
                                   f"Skills: {req.skills}, Languages: {req.languages}. "
                                   f"Check that the prompts file has matching categories."})
                return
            actual_prompts = filtered_path
            yield _sse({"type": "log", "stage": "generate",
                        "text": f"Curriculum: {kept_n}/{orig_n} prompts match skills "
                                f"{req.skills}" + (f" langs {req.languages}" if req.languages else "")})

        # ── Stages 1-3 (slow path) ─────────────────────────────────────────
        # Only the traditional teacher-generation flow runs these. Speed-Run
        # already wrote consensus_sft.jsonl + purification_report.json above
        # and set gold_n / silver_n / drop_n / n_prompts / n_teachers.
        if not speed_run:
            # ── Stage 1: Generate ──────────────────────────────────────────
            n_prompts = _safe_prompt_count(actual_prompts)
            n_teachers = len(req.teachers)

            # Persist curriculum so downstream stages and dashboards know what was trained
            if req.skills:
                curriculum = {
                    "skills": req.skills,
                    "languages": req.languages,
                    "code_langs": req.code_langs,
                    "filtered_prompts": _safe_prompt_count(actual_prompts),
                    "original_prompts": _safe_prompt_count(req.prompts_path),
                }
                (saves_dir / "curriculum.json").write_text(
                    json.dumps(curriculum, indent=2), encoding="utf-8",
                )

            def _responses_ok() -> bool:
                if not Path(out_jsonl).is_file():
                    return False
                try:
                    n_rows = sum(1 for l in open(out_jsonl, encoding="utf-8") if l.strip())
                    return n_rows >= n_teachers * n_prompts * 0.5
                except OSError:
                    return False

            def _responses_have_content() -> bool:
                """Check that teacher responses actually contain non-empty answers."""
                if not Path(out_jsonl).is_file():
                    return False
                try:
                    non_empty = 0
                    total = 0
                    for raw in open(out_jsonl, encoding="utf-8"):
                        raw = raw.strip()
                        if not raw:
                            continue
                        row = json.loads(raw)
                        for tname, tdata in row.get("teachers", {}).items():
                            total += 1
                            ans = tdata.get("answer", "").strip()
                            if ans:
                                non_empty += 1
                    return total > 0 and (non_empty / total) > 0.05
                except (OSError, json.JSONDecodeError, KeyError):
                    return False

            if _responses_ok():
                yield _ev("generate", "skip", "teacher_responses.jsonl exists -- skipping.")
            else:
                yield _ev("generate", "running", "Building teacher manifest...")
                # Write manifest with hardware-aware acceleration settings
                manifest_path = out / "teacher_manifest.json"
                teachers_data = [{"name": Path(t).stem, "gguf": t} for t in req.teachers]
                n_teachers = max(len(teachers_data), 1)
                # Auto-detect optimal inference config via zen_core_libs
                try:
                    from zen_core_libs.common.device import auto_configure_inference
                    hw_config = auto_configure_inference(num_concurrent_models=n_teachers)
                    llama_cpp_accel: dict[str, Any] = {
                        k: v for k, v in hw_config.items() if not k.startswith("_")
                    }
                    gpu_info = hw_config.get("_note", "auto-detected")
                except ImportError:
                    cpu_count = os.cpu_count() or 8
                    threads_per_model = max(2, (cpu_count - 2) // n_teachers)
                    llama_cpp_accel = {
                        "n_threads": threads_per_model,
                        "n_batch": 512,
                        "n_ubatch": 512,
                    }
                    gpu_info = f"CPU-only ({cpu_count} cores, {threads_per_model} threads/model)"
                manifest_data = {
                    "backend": req.backend,
                    "teachers": teachers_data,
                    "max_models": n_teachers,
                    "n_ctx": 4096,
                    "llama_cpp": llama_cpp_accel,
                }
                manifest_path.write_text(
                    json.dumps(manifest_data, indent=2),
                    encoding="utf-8",
                )
                yield _ev("generate", "running", f"Manifest: {n_teachers} teachers, {gpu_info}")

                cmd_gen = [
                    PYTHON, str(SCRIPTS_DIR / "multi_teacher_generate.py"),
                    "--manifest", str(manifest_path),
                    "--prompts", actual_prompts,
                    "--out", out_jsonl,
                    "--max-tokens", str(int(req.max_tokens)),
                    "--temperature", str(req.temperature),
                    "--dispatch-mode", "teacher-fifo",
                    "--fifo-size", "0",
                    "--adaptive-budgets",
                ]
                if req.profile_path and Path(req.profile_path).is_file():
                    cmd_gen.extend(["--profile", req.profile_path])

                yield _ev("generate", "running", "Expert teachers generating responses...")
                proc = _popen(cmd_gen)
                last_eta_check = 0.0
                gen_start_time = time.time()
                prev_done = 0
                # Suppress known-harmless llama_cpp cleanup noise (AttributeError in __del__)
                _noise = (
                    "object has no attribute 'sampler'",
                    "Exception ignored in:",
                    "in __del__",
                    "self.close()",
                    "if self.sampler is not None",
                    "^^^^^^^^^^",
                )
                loop = asyncio.get_event_loop()
                assert proc.stdout is not None
                while proc.poll() is None:
                    line = await loop.run_in_executor(None, proc.stdout.readline)
                    if line:
                        stripped = line.rstrip()
                        if not any(n in stripped for n in _noise):
                            yield _sse({"type": "log", "stage": "generate", "text": stripped})
                    now = time.time()
                    if now - last_eta_check > 5:
                        last_eta_check = now
                        # Quick ETA from checkpoint progress
                        ckpt_dir = out / "checkpoints"
                        done_total = 0
                        elapsed_samples: list[float] = []
                        if ckpt_dir.is_dir():
                            for tp in req.teachers:
                                ckpt = ckpt_dir / f"{_sanitize(Path(tp).stem)}.jsonl"
                                if ckpt.is_file():
                                    try:
                                        for ckline in open(ckpt, encoding="utf-8"):
                                            ckline = ckline.strip()
                                            if not ckline:
                                                continue
                                            done_total += 1
                                            try:
                                                rec = json.loads(ckline)
                                                es = rec.get("response", {}).get("elapsed_s")
                                                if isinstance(es, (int, float)) and es > 0:
                                                    elapsed_samples.append(float(es))
                                            except (json.JSONDecodeError, KeyError, ValueError):
                                                pass
                                    except OSError:
                                        pass
                        tgt = n_teachers * n_prompts
                        pct = round(100.0 * done_total / tgt, 1) if tgt else 0
                        wall_elapsed = now - gen_start_time
                        # ETA from per-response elapsed times (trimmed mean)
                        eta_s: float | None = None
                        rate_str = ""
                        if elapsed_samples and tgt > done_total:
                            trimmed = sorted(elapsed_samples)[:max(1, int(len(elapsed_samples) * 0.9))]
                            avg_s = sum(trimmed) / len(trimmed)
                            eta_s = round((tgt - done_total) * avg_s)
                        # Fallback: wall-clock rate
                        if eta_s is None and done_total > 0 and tgt > done_total:
                            wall_per = wall_elapsed / done_total
                            eta_s = round((tgt - done_total) * wall_per)
                        if done_total > 0 and wall_elapsed > 0:
                            rps = done_total / wall_elapsed
                            if rps >= 1:
                                rate_str = f"{rps:.1f} resp/s"
                            else:
                                rate_str = f"{1/rps:.1f} s/resp"
                        # Format elapsed
                        def _fmt_hms(secs: float) -> str:
                            h = int(secs) // 3600
                            m = (int(secs) % 3600) // 60
                            s = int(secs) % 60
                            return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"
                        elapsed_str = _fmt_hms(wall_elapsed)
                        eta_str = _fmt_hms(eta_s) if eta_s else ""
                        msg = f"Progress: {done_total}/{tgt} responses ({pct}%)"
                        yield _ev("generate", "running", msg, pct=pct,
                                  done=done_total, total=tgt,
                                  elapsed=elapsed_str, eta=eta_str,
                                  rate=rate_str)
                rc = proc.wait()
                if rc != 0:
                    yield _ev("generate", "fail", f"Generation failed (exit {rc}).")
                    yield _sse({"type": "error", "msg": "Generation failed."})
                    return
                yield _ev("generate", "done", f"Generation complete -- {n_teachers * n_prompts} responses.")

            # Post-generation validation: check answers aren't empty
            if not _responses_have_content():
                yield _ev("generate", "fail",
                          "Teachers produced EMPTY answers! Check that teacher GGUF models are valid "
                          "and the inference backend is working. All responses have blank \"answer\" fields.")
                yield _sse({"type": "error", "msg":
                            "Generation failed: all teacher answers are empty. "
                            "The teachers ran but produced no text. Verify your GGUF models work."})
                return

            # ── Stage 2: Hallucination gates ────────────────────────────────────
            halluc_rpt = out / "hallucination_report.json"
            if not req.enable_halluc:
                yield _ev("halluc", "skip", "Disabled.")
            elif halluc_rpt.is_file():
                yield _ev("halluc", "skip", "Report already exists -- skipping.")
            else:
                yield _ev("halluc", "running", "Running 5-gate hallucination pipeline...")
                try:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, lambda: _run_halluc_gates_sync(
                        out_jsonl, req.zena_gguf, req.output_dir
                    ))
                    yield _ev("halluc", "done", "Gates complete.")
                except Exception as exc:
                    yield _ev("halluc", "fail", f"Gates error: {exc}")

            # ── Stage 3: Purify ─────────────────────────────────────────────────
            purify_rpt = out / "purification_report.json"
            gold_n = silver_n = drop_n = 0
            if purify_rpt.is_file():
                try:
                    with open(purify_rpt, encoding="utf-8") as f:
                        rpt = json.load(f)
                except (json.JSONDecodeError, OSError) as exc:
                    yield _ev("purify", "fail", f"Corrupt purification report: {exc}")
                    return
                gold_n = rpt.get("gold_count", 0)
                silver_n = rpt.get("silver_count", 0)
                drop_n = rpt.get("dropped_count", 0)
                yield _ev("purify", "skip", f"Already purified -- GOLD {gold_n} / SILVER {silver_n} / DROP {drop_n}",
                          gold=gold_n, silver=silver_n, drop=drop_n)
            else:
                yield _ev("purify", "running", "Purifying teacher responses...")
                cmd_p = [
                    PYTHON, str(SCRIPTS_DIR / "purify_teacher_outputs.py"),
                    "--input", out_jsonl,
                    "--out-dir", req.output_dir,
                    "--answer-threshold", str(req.answer_th),
                    "--reason-threshold", str(req.reason_th),
                ]
                proc_p = _popen(cmd_p)
                loop = asyncio.get_event_loop()
                assert proc_p.stdout is not None
                while proc_p.poll() is None:
                    line = await loop.run_in_executor(None, proc_p.stdout.readline)
                    if line:
                        yield _sse({"type": "log", "stage": "purify", "text": line.rstrip()})
                rc = proc_p.wait()
                if rc != 0:
                    yield _ev("purify", "fail", "Purification failed.")
                    return
                if purify_rpt.is_file():
                    try:
                        with open(purify_rpt, encoding="utf-8") as f:
                            rpt = json.load(f)
                    except (json.JSONDecodeError, OSError):
                        rpt = {}
                    gold_n = rpt.get("gold_count", 0)
                    silver_n = rpt.get("silver_count", 0)
                    drop_n = rpt.get("dropped_count", 0)
                yield _ev("purify", "done", f"GOLD {gold_n}  SILVER {silver_n}  DROP {drop_n}",
                          gold=gold_n, silver=silver_n, drop=drop_n)

        if not req.do_train:
            yield _ev("configs", "skip", "Training skipped.")
            yield _ev("sft", "skip"); yield _ev("dpo", "skip")
            yield _ev("merge", "skip"); yield _ev("gguf", "skip")
            yield _ev("eval", "skip"); yield _ev("dashboard", "skip")
            yield _sse({"type": "done_all", "gold": gold_n, "silver": silver_n, "drop": drop_n})
            return

        # Guard: cannot train with zero GOLD data
        if gold_n == 0:
            yield _ev("configs", "fail",
                      "Cannot train: GOLD (consensus) count is 0. "
                      "All teacher responses were dropped. Check that teachers "
                      "actually produced answers (not empty strings). "
                      "Try lowering Gold/Silver thresholds, or re-run generation.")
            yield _ev("sft", "skip"); yield _ev("dpo", "skip")
            yield _ev("merge", "skip"); yield _ev("gguf", "skip")
            yield _ev("eval", "skip"); yield _ev("dashboard", "skip")
            yield _sse({"type": "done_all", "gold": gold_n, "silver": silver_n, "drop": drop_n})
            return

        # ── Sequential batch: normalize students into a list ───────────────
        # When students_batch is populated the Studio is running multi-student
        # training. Stages 1-3 (generate/halluc/purify) already ran ONCE above,
        # producing a single GOLD/SILVER dataset that every student in the
        # batch shares. The loop below re-runs stages 4-11 once per student
        # with a unique saves tag so adapters/merged/gguf don't collide.
        if req.students_batch:
            batch = [
                {
                    "name":     (s.get("name") or f"student_{i+1}").strip(),
                    "hf_id":    (s.get("hf_id") or s.get("student_model") or req.student_model).strip(),
                    "cpu_safe": bool(s.get("cpu_safe", req.cpu_safe)),
                }
                for i, s in enumerate(req.students_batch)
            ]
        else:
            batch = [{
                "name":     req.saves_tag or "student",
                "hf_id":    req.student_model,
                "cpu_safe": req.cpu_safe,
            }]

        report_text = ""
        last_student_status = "done"

        for _s_idx, _s in enumerate(batch):
            # Honor STOP between students -- don't start the next one.
            if _is_cancel_requested():
                yield _sse({"type": "log", "stage": "batch",
                            "text": f"Batch cancelled before student {_s_idx+1}/{len(batch)}."})
                last_student_status = "cancelled"
                break

            # Per-student saves tag + dir. For a single student we reuse the
            # caller's tag verbatim so existing runs/flags stay compatible.
            if len(batch) > 1:
                _safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", _s["name"]).strip("_") or f"s{_s_idx+1}"
                s_tag = f"{tag}_{_safe_name}"
            else:
                s_tag = tag
            s_saves_dir = ROOT_DIR / "saves" / s_tag
            s_saves_dir.mkdir(parents=True, exist_ok=True)
            _set_metrics_sink(s_saves_dir)

            # Drop a curriculum sidecar next to each student's saves so
            # postmortem_agent.py can report per-student context.
            if req.skills:
                (s_saves_dir / "curriculum.json").write_text(
                    json.dumps({
                        "skills": req.skills,
                        "languages": req.languages,
                        "code_langs": req.code_langs,
                        "student_name": _s["name"],
                        "student_model": _s["hf_id"],
                    }, indent=2),
                    encoding="utf-8",
                )

            def _student_end(status: str) -> str:
                return _sse({
                    "type": "student_end",
                    "idx": _s_idx, "total": len(batch),
                    "name": _s["name"], "tag": s_tag, "status": status,
                })

            yield _sse({
                "type": "student_begin",
                "idx": _s_idx, "total": len(batch),
                "name": _s["name"], "hf_id": _s["hf_id"], "tag": s_tag,
            })

            # ── Stage 4: Gen configs ────────────────────────────────────────
            sft_yaml = config_out / f"{s_tag}_sft.yaml"
            dpo_yaml = config_out / f"{s_tag}_dpo.yaml"
            if sft_yaml.is_file():
                yield _ev("configs", "skip", "Configs exist -- skipping.")
            else:
                yield _ev("configs", "running", f"Generating configs for {_s['hf_id']}...")
                cmd_cfg = [
                    PYTHON, str(SCRIPTS_DIR / "gen_distill_configs.py"),
                    "--student", _s["hf_id"],
                    "--data-dir", req.output_dir,
                    "--out-dir", str(config_out),
                    "--tag", s_tag,
                ]
                if _s["cpu_safe"]:
                    cmd_cfg.append("--cpu-safe")
                rc, _, err = _run_script(cmd_cfg, timeout=120)
                if rc != 0:
                    yield _ev("configs", "fail", err[-300:])
                    last_student_status = "fail"
                    yield _student_end("fail")
                    break
                # Verify the SFT yaml was actually created
                if not sft_yaml.is_file():
                    yield _ev("configs", "fail",
                              f"Config generation succeeded but {sft_yaml.name} was not created. "
                              f"This usually means consensus_sft.jsonl is missing or empty (GOLD=0). "
                              f"Check purification output.")
                    last_student_status = "fail"
                    yield _student_end("fail")
                    break
                yield _ev("configs", "done", "Configs ready.")

            # ── Stage 5: SFT ────────────────────────────────────────────────
            sft_flag = s_saves_dir / "sft_complete.flag"
            if sft_flag.is_file():
                yield _ev("sft", "skip", "SFT already done.")
            else:
                yield _ev("sft", "running", "SFT training started (may take 20-120 min)...")
                proc_sft = _popen([sys.executable, "-m", "llamafactory.cli", "train", str(sft_yaml)])
                loop = asyncio.get_event_loop()
                assert proc_sft.stdout is not None
                _last_metrics_emit = 0.0
                while proc_sft.poll() is None:
                    line = await loop.run_in_executor(None, proc_sft.stdout.readline)
                    if line:
                        yield _sse({"type": "log", "stage": "sft", "text": line.rstrip()})
                        prog = _parse_train_progress(line)
                        if prog:
                            yield _sse({"type": "train_progress", "stage": "sft", **prog})
                    # Emit a metrics frame at most once per second so the rail
                    # stays alive even when training stdout is silent.
                    now_ts = time.time()
                    if now_ts - _last_metrics_emit >= 1.0:
                        frame = _metrics_frame()
                        if frame:
                            yield frame
                        _last_metrics_emit = now_ts
                rc = proc_sft.wait()
                if rc != 0:
                    cancelled = _is_cancel_requested()
                    yield _ev("sft", "fail",
                              "SFT cancelled by user." if cancelled else f"SFT failed (exit {rc}).")
                    last_student_status = "cancelled" if cancelled else "fail"
                    yield _student_end(last_student_status)
                    break
                sft_flag.touch()
                yield _ev("sft", "done", "SFT complete.")

            # ── Stage 6: DPO ────────────────────────────────────────────────
            dpo_flag = s_saves_dir / "dpo_complete.flag"
            if not dpo_yaml.is_file():
                yield _ev("dpo", "skip", "No DPO config (no SILVER data).")
            elif dpo_flag.is_file():
                yield _ev("dpo", "skip", "DPO already done.")
            else:
                yield _ev("dpo", "running", "DPO training...")
                proc_dpo = _popen([sys.executable, "-m", "llamafactory.cli", "train", str(dpo_yaml)])
                loop = asyncio.get_event_loop()
                assert proc_dpo.stdout is not None
                _last_metrics_emit = 0.0
                while proc_dpo.poll() is None:
                    line = await loop.run_in_executor(None, proc_dpo.stdout.readline)
                    if line:
                        yield _sse({"type": "log", "stage": "dpo", "text": line.rstrip()})
                        prog = _parse_train_progress(line)
                        if prog:
                            yield _sse({"type": "train_progress", "stage": "dpo", **prog})
                    now_ts = time.time()
                    if now_ts - _last_metrics_emit >= 1.0:
                        frame = _metrics_frame()
                        if frame:
                            yield frame
                        _last_metrics_emit = now_ts
                rc = proc_dpo.wait()
                if rc != 0:
                    cancelled = _is_cancel_requested()
                    yield _ev("dpo", "fail",
                              "DPO cancelled by user." if cancelled else f"DPO failed (exit {rc}).")
                    last_student_status = "cancelled" if cancelled else "fail"
                    yield _student_end(last_student_status)
                    break
                dpo_flag.touch()
                yield _ev("dpo", "done", "DPO complete.")

            # ── Stage 7: Merge ──────────────────────────────────────────────
            merge_yaml = config_out / f"{s_tag}_merge.yaml"
            merged_dir = s_saves_dir / "merged"
            if (merged_dir / "config.json").is_file():
                yield _ev("merge", "skip", "Merged model exists.")
            elif not merge_yaml.is_file():
                yield _ev("merge", "skip", "No merge config.")
            else:
                yield _ev("merge", "running", "Merging LoRA adapters...")
                proc_m = _popen([sys.executable, "-m", "llamafactory.cli", "export", str(merge_yaml)])
                loop = asyncio.get_event_loop()
                assert proc_m.stdout is not None
                while proc_m.poll() is None:
                    line = await loop.run_in_executor(None, proc_m.stdout.readline)
                    if line:
                        yield _sse({"type": "log", "stage": "merge", "text": line.rstrip()})
                rc = proc_m.wait()
                if rc != 0:
                    cancelled = _is_cancel_requested()
                    yield _ev("merge", "fail",
                              "Merge cancelled by user." if cancelled else f"Merge failed (exit {rc}).")
                    last_student_status = "cancelled" if cancelled else "fail"
                    yield _student_end(last_student_status)
                    break
                yield _ev("merge", "done", "Merge complete.")

            # ── Stage 8: GGUF ───────────────────────────────────────────────
            gguf_dir = s_saves_dir / "gguf"
            if (gguf_dir / "slim_down_results.jsonl").is_file():
                yield _ev("gguf", "skip", "GGUF already exported.")
            elif not merged_dir.is_dir():
                yield _ev("gguf", "skip", "No merged model -- skipping GGUF.")
            else:
                yield _ev("gguf", "running", "Converting to GGUF Q4_K_M...")
                gguf_dir.mkdir(parents=True, exist_ok=True)
                cmd_gguf = [
                    PYTHON, str(SCRIPTS_DIR / "slim_down.py"),
                    "--model-dir", str(merged_dir),
                    "--out-dir", str(gguf_dir),
                    "--tag", s_tag,
                    "--quant", "q4_k_m",
                ]
                proc_g = _popen(cmd_gguf)
                loop = asyncio.get_event_loop()
                assert proc_g.stdout is not None
                while proc_g.poll() is None:
                    line = await loop.run_in_executor(None, proc_g.stdout.readline)
                    if line:
                        yield _sse({"type": "log", "stage": "gguf", "text": line.rstrip()})
                rc = proc_g.wait()
                if rc != 0:
                    yield _ev("gguf", "fail", f"GGUF export failed (exit {rc}).")
                else:
                    yield _ev("gguf", "done", "GGUF export complete.")

            # ── Stage 9: Eval ───────────────────────────────────────────────
            probes_path = next(
                (str(p) for p in [
                    out / "eval_probes.jsonl",
                    out / "purified" / "eval_probes.jsonl",
                    Path(req.prompts_path),
                ] if Path(str(p)).is_file()),
                None,
            )
            if not req.run_eval:
                yield _ev("eval", "skip", "Disabled in Studio.")
                yield _ev("dashboard", "skip", "Eval disabled -- no report.")
            elif (s_saves_dir / "eval_scorecards.jsonl").is_file():
                yield _ev("eval", "skip", "Scorecards exist.")
            elif probes_path is None:
                yield _ev("eval", "skip", "No eval_probes.jsonl.")
            else:
                yield _ev("eval", "running", "Running graduation exam...")
                cmd_ev = [
                    PYTHON, str(SCRIPTS_DIR / "eval_student_panel.py"),
                    "--saves-tag", s_tag,
                    "--probes", probes_path,
                ]
                proc_ev = _popen(cmd_ev)
                loop = asyncio.get_event_loop()
                assert proc_ev.stdout is not None
                while proc_ev.poll() is None:
                    line = await loop.run_in_executor(None, proc_ev.stdout.readline)
                    if line:
                        yield _sse({"type": "log", "stage": "eval", "text": line.rstrip()})
                rc = proc_ev.wait()
                if rc != 0:
                    yield _ev("eval", "fail", f"Exam failed (exit {rc}).")
                else:
                    yield _ev("eval", "done", "Exam complete.")

            # ── Stage 10: Dashboard ─────────────────────────────────────────
            if not req.run_eval:
                pass  # dashboard already reported as skipped above
            else:
                yield _ev("dashboard", "running", "Generating graduation report...")
                rpt_md = s_saves_dir / "graduation_report.md"
                rc, _, err = _run_script([
                    PYTHON, str(SCRIPTS_DIR / "graduation_dashboard.py"),
                    "--saves-tag", s_tag,
                    "--export-markdown", str(rpt_md),
                ], timeout=120)
                if rc == 0 and rpt_md.is_file():
                    student_report = rpt_md.read_text(encoding="utf-8")
                    if len(batch) > 1:
                        report_text += (
                            f"\n\n## Student {_s_idx+1}/{len(batch)}: "
                            f"{_s['name']}\n\n{student_report}"
                        )
                    else:
                        report_text = student_report
                    yield _ev("dashboard", "done", f"Report: {rpt_md}")
                else:
                    yield _ev("dashboard", "fail", err[-200:])

            # ── Stage 11: Postmortem ────────────────────────────────────────
            # Auto-run the postmortem agent over the just-completed student
            # so the next iteration can pick up YAML auto-tune suggestions.
            if not req.run_postmortem:
                yield _ev("postmortem", "skip", "Disabled in Studio.")
            else:
                try:
                    postmortem_dir = ROOT_DIR / "docs" / "postmortem"
                    postmortem_dir.mkdir(parents=True, exist_ok=True)
                    rc_pm, _, err_pm = _run_script(
                        [PYTHON, str(SCRIPTS_DIR / "postmortem_agent.py"),
                         str(s_saves_dir), "--report-dir", str(postmortem_dir)],
                        timeout=120,
                    )
                    if rc_pm == 0:
                        pm_path = postmortem_dir / f"RUN_{s_saves_dir.name}.md"
                        yield _ev("postmortem", "done", f"Recommendations: {pm_path}")
                    else:
                        yield _ev("postmortem", "fail", err_pm[-200:] if err_pm else "postmortem failed")
                except Exception as exc:  # noqa: BLE001
                    yield _ev("postmortem", "fail", f"postmortem error: {exc}")

            # Student completed successfully
            last_student_status = "done"
            yield _student_end("done")

        # Stop streaming metrics into the run sidecar -- the batch is over.
        _set_metrics_sink(None)
        _set_metrics_stage("idle")

        yield _sse({
            "type": "done_all",
            "gold": gold_n, "silver": silver_n, "drop": drop_n,
            "report": report_text,
            "batch_size": len(batch),
            "batch_status": last_student_status,
        })

    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.post("/api/pipeline/stop")
async def pipeline_stop() -> JSONResponse:
    """Cancel the currently running pipeline.

    Flips the cancel flag and kills the active subprocess (if any). Stages
    check the flag between iterations and bail out cleanly. Returns a small
    summary so the UI can show "nothing to stop" when idle.
    """
    summary = _request_cancel()
    return JSONResponse(summary)


# ---------------------------------------------------------------------------
# Hallucination gates (sync, runs in executor)
# ---------------------------------------------------------------------------

def _run_halluc_gates_sync(responses_jsonl: str, zena_gguf: str, output_dir: str) -> None:
    try:
        from hallucination_gates import GateVerdict, run_hallucination_pipeline
    except ImportError:
        return

    dean = _get_dean(zena_gguf) if zena_gguf and zena_gguf.strip() else None
    total = passed = flagged = failed = 0
    all_reports = []

    with open(responses_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                row = {}
            prompt = row.get("prompt", "")
            teachers_data = row.get("teachers", {})
            teacher_answers = {k: v.get("answer", v.get("raw", "")) for k, v in teachers_data.items()}
            if not teacher_answers:
                continue
            reports = run_hallucination_pipeline(
                prompt=prompt,
                teacher_answers=teacher_answers,
                zena_dean=dean,
                borderline_only=True,
            )
            for name, report in reports.items():
                total += 1
                if report.final_verdict == GateVerdict.PASS:
                    passed += 1
                elif report.final_verdict == GateVerdict.FLAG:
                    flagged += 1
                else:
                    failed += 1
                all_reports.append(report)

    rpt_out = Path(output_dir) / "hallucination_report.json"
    with open(rpt_out, "w", encoding="utf-8") as f:
        json.dump([
            {"teacher": r.teacher, "prompt_id": r.prompt_id,
             "verdict": r.final_verdict.value, "score": r.final_score,
             "gates": [{"gate": g.gate, "verdict": g.verdict.value, "score": g.score}
                       for g in r.gate_results]}
            for r in all_reports
        ], f, indent=2)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="University of Distillation server")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7870")))
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    # Install process guard FIRST so we sweep any orphans from a previous
    # crashed run before we start spawning new children. Idempotent.
    if _PROCESS_GUARD_OK and process_guard is not None:
        swept = process_guard.install()
        if swept:
            print(f"[process_guard] startup sweep killed {swept} orphan process(es)")  # xray: ignore[PY-004]
    else:
        print("[process_guard] not available — child crashes may leave orphans. Install psutil.")  # xray: ignore[PY-004]

    print(f"Starting server on http://{args.host}:{args.port}")  # xray: ignore[PY-004]
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
