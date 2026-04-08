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
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(ROOT_DIR),
        env={**os.environ, "PYTHONUTF8": "1"},
    )


def _collect(proc: subprocess.Popen, timeout: int = 86400) -> tuple[int, str]:
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
    return proc.returncode, out or ""


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------

def _sse(data: Any) -> str:
    """Encode one SSE data frame."""
    return f"data: {json.dumps(data)}\n\n"


async def _stream_proc(proc: subprocess.Popen) -> AsyncGenerator[str, None]:
    """Stream subprocess stdout line-by-line as SSE, then emit exit code."""
    loop = asyncio.get_event_loop()
    assert proc.stdout is not None
    while True:
        line = await loop.run_in_executor(None, proc.stdout.readline)
        if not line:
            break
        yield _sse({"type": "log", "text": line.rstrip()})
    rc = proc.wait()
    yield _sse({"type": "done", "rc": rc})


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
    # Curriculum: skills & languages the student enrolled for
    skills: list[str] = []       # e.g. ["translation", "ocr_cleanup"]
    languages: list[str] = []    # e.g. ["en", "de", "fr"]
    code_langs: list[str] = []   # e.g. ["python", "rust"]


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
            return _sse({"type": "stage", "stage": stage, "status": status, "msg": msg, **extra})

        out = Path(req.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        out_jsonl = str(out / "teacher_responses.jsonl")
        tag = req.saves_tag.strip() or "distill_auto"
        saves_dir = ROOT_DIR / "saves" / tag
        saves_dir.mkdir(parents=True, exist_ok=True)
        config_out = ROOT_DIR / "examples" / "distillation" / "auto"

        if not req.teachers:
            yield _sse({"type": "error", "msg": "No teachers listed."})
            return
        if not req.prompts_path or not Path(req.prompts_path).is_file():
            yield _sse({"type": "error", "msg": f"Prompts file not found: {req.prompts_path}"})
            return

        # ── Curriculum filter: keep only prompts matching enrolled skills ──
        actual_prompts = req.prompts_path
        if req.skills:
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

        # ── Stage 1: Generate ──────────────────────────────────────────────
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
            # Write manifest
            manifest_path = out / "teacher_manifest.json"
            teachers_data = [{"name": Path(t).stem, "gguf": t} for t in req.teachers]
            manifest_path.write_text(
                json.dumps({"backend": req.backend, "teachers": teachers_data}, indent=2),
                encoding="utf-8",
            )

            cmd_gen = [
                PYTHON, str(SCRIPTS_DIR / "multi_teacher_generate.py"),
                "--manifest", str(manifest_path),
                "--prompts", actual_prompts,
                "--out", out_jsonl,
                "--max-tokens", str(int(req.max_tokens)),
                "--temperature", str(req.temperature),
            ]
            if req.profile_path and Path(req.profile_path).is_file():
                cmd_gen.extend(["--profile", req.profile_path])

            yield _ev("generate", "running", "Expert teachers generating responses...")
            proc = _popen(cmd_gen)
            last_eta_check = 0.0
            loop = asyncio.get_event_loop()
            assert proc.stdout is not None
            while proc.poll() is None:
                line = await loop.run_in_executor(None, proc.stdout.readline)
                if line:
                    yield _sse({"type": "log", "stage": "generate", "text": line.rstrip()})
                now = time.time()
                if now - last_eta_check > 8:
                    last_eta_check = now
                    # Quick ETA from checkpoint progress
                    ckpt_dir = out / "checkpoints"
                    done_total = 0
                    if ckpt_dir.is_dir():
                        for tp in req.teachers:
                            ckpt = ckpt_dir / f"{_sanitize(Path(tp).stem)}.jsonl"
                            if ckpt.is_file():
                                try:
                                    done_total += sum(1 for l in open(ckpt, encoding="utf-8") if l.strip())
                                except OSError:
                                    pass
                    tgt = n_teachers * n_prompts
                    pct = round(100.0 * done_total / tgt, 1) if tgt else 0
                    yield _ev("generate", "running", f"Progress: {done_total}/{tgt} responses ({pct}%)", pct=pct)
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

        # ── Stage 4: Gen configs ────────────────────────────────────────────
        sft_yaml = config_out / "distill_auto_sft.yaml"
        dpo_yaml = config_out / "distill_auto_dpo.yaml"
        if sft_yaml.is_file():
            yield _ev("configs", "skip", "Configs exist -- skipping.")
        else:
            yield _ev("configs", "running", f"Generating configs for {req.student_model}...")
            cmd_cfg = [
                PYTHON, str(SCRIPTS_DIR / "gen_distill_configs.py"),
                "--student", req.student_model,
                "--data-dir", req.output_dir,
                "--out-dir", str(config_out),
                "--tag", "distill_auto",
            ]
            if req.cpu_safe:
                cmd_cfg.append("--cpu-safe")
            rc, _, err = _run_script(cmd_cfg, timeout=120)
            if rc != 0:
                yield _ev("configs", "fail", err[-300:])
                return
            # Verify the SFT yaml was actually created
            if not sft_yaml.is_file():
                yield _ev("configs", "fail",
                          f"Config generation succeeded but {sft_yaml.name} was not created. "
                          f"This usually means consensus_sft.jsonl is missing or empty (GOLD=0). "
                          f"Check purification output.")
                return
            yield _ev("configs", "done", "Configs ready.")

        # ── Stage 5: SFT ────────────────────────────────────────────────────
        sft_flag = saves_dir / "sft_complete.flag"
        if sft_flag.is_file():
            yield _ev("sft", "skip", "SFT already done.")
        else:
            yield _ev("sft", "running", "SFT training started (may take 20-120 min)...")
            proc_sft = _popen([sys.executable, "-m", "llamafactory.cli", "train", str(sft_yaml)])
            loop = asyncio.get_event_loop()
            assert proc_sft.stdout is not None
            while proc_sft.poll() is None:
                line = await loop.run_in_executor(None, proc_sft.stdout.readline)
                if line:
                    yield _sse({"type": "log", "stage": "sft", "text": line.rstrip()})
                    prog = _parse_train_progress(line)
                    if prog:
                        yield _sse({"type": "train_progress", "stage": "sft", **prog})
            rc = proc_sft.wait()
            if rc != 0:
                yield _ev("sft", "fail", f"SFT failed (exit {rc}).")
                return
            sft_flag.touch()
            yield _ev("sft", "done", "SFT complete.")

        # ── Stage 6: DPO ────────────────────────────────────────────────────
        dpo_flag = saves_dir / "dpo_complete.flag"
        if not dpo_yaml.is_file():
            yield _ev("dpo", "skip", "No DPO config (no SILVER data).")
        elif dpo_flag.is_file():
            yield _ev("dpo", "skip", "DPO already done.")
        else:
            yield _ev("dpo", "running", "DPO training...")
            proc_dpo = _popen([sys.executable, "-m", "llamafactory.cli", "train", str(dpo_yaml)])
            loop = asyncio.get_event_loop()
            assert proc_dpo.stdout is not None
            while proc_dpo.poll() is None:
                line = await loop.run_in_executor(None, proc_dpo.stdout.readline)
                if line:
                    yield _sse({"type": "log", "stage": "dpo", "text": line.rstrip()})
                    prog = _parse_train_progress(line)
                    if prog:
                        yield _sse({"type": "train_progress", "stage": "dpo", **prog})
            rc = proc_dpo.wait()
            if rc != 0:
                yield _ev("dpo", "fail", f"DPO failed (exit {rc}).")
                return
            dpo_flag.touch()
            yield _ev("dpo", "done", "DPO complete.")

        # ── Stage 7: Merge ───────────────────────────────────────────────────
        merge_yaml = config_out / "distill_auto_merge.yaml"
        merged_dir = saves_dir / "merged"
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
                yield _ev("merge", "fail", f"Merge failed (exit {rc}).")
                return
            yield _ev("merge", "done", "Merge complete.")

        # ── Stage 8: GGUF ────────────────────────────────────────────────────
        gguf_dir = saves_dir / "gguf"
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
                "--tag", tag,
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

        # ── Stage 9: Eval ────────────────────────────────────────────────────
        probes_path = next(
            (str(p) for p in [
                out / "eval_probes.jsonl",
                out / "purified" / "eval_probes.jsonl",
                Path(req.prompts_path),
            ] if Path(str(p)).is_file()),
            None,
        )
        if (saves_dir / "eval_scorecards.jsonl").is_file():
            yield _ev("eval", "skip", "Scorecards exist.")
        elif probes_path is None:
            yield _ev("eval", "skip", "No eval_probes.jsonl.")
        else:
            yield _ev("eval", "running", "Running graduation exam...")
            cmd_ev = [
                PYTHON, str(SCRIPTS_DIR / "eval_student_panel.py"),
                "--saves-tag", tag,
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

        # ── Stage 10: Dashboard ──────────────────────────────────────────────
        yield _ev("dashboard", "running", "Generating graduation report...")
        rpt_md = saves_dir / "graduation_report.md"
        rc, _, err = _run_script([
            PYTHON, str(SCRIPTS_DIR / "graduation_dashboard.py"),
            "--saves-tag", tag,
            "--export-markdown", str(rpt_md),
        ], timeout=120)
        report_text = ""
        if rc == 0 and rpt_md.is_file():
            report_text = rpt_md.read_text(encoding="utf-8")
            yield _ev("dashboard", "done", f"Report: {rpt_md}")
        else:
            yield _ev("dashboard", "fail", err[-200:])

        yield _sse({
            "type": "done_all",
            "gold": gold_n, "silver": silver_n, "drop": drop_n,
            "report": report_text,
        })

    return StreamingResponse(_gen(), media_type="text/event-stream")


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
    print(f"Starting server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
