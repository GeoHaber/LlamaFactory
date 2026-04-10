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

"""Process guard — kill child process trees on exit, recover orphans on startup.

Solves the "56 GB orphaned multi_teacher_generate.py" problem: when
distill_server.py is killed (or crashes), its subprocess.Popen children
were left running, eating RAM until the user noticed.

What this module does:
  1. Tracks every child process spawned by the parent in a dict.
  2. On normal exit (atexit) OR signal (SIGINT/SIGTERM/SIGBREAK/SIGHUP),
     walks every registered child + ALL its descendants and force-kills them.
  3. Persists the live PID set to runtime/active_pids.json so even SIGKILL
     of the parent leaves a recovery file behind.
  4. On startup, sweep_orphans() reads that recovery file and kills any
     leftover python processes from a previous crash (carefully — only
     processes that look like they belong to this codebase).

Usage:
    import process_guard
    process_guard.install()  # at top of main()

    proc = subprocess.Popen([...])
    process_guard.register_child(proc.pid, "multi_teacher_generate")
    try:
        rc = proc.wait()
    finally:
        process_guard.unregister_child(proc.pid)

If psutil isn't installed, falls back to plain os.kill() — works for the
direct child but can't walk the tree. Install psutil for the full effect.
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import sys
import threading
from pathlib import Path

try:
    import psutil  # xray: ignore[SEC-015]
    _PSUTIL_OK = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    _PSUTIL_OK = False


# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
GUARD_DIR = ROOT / "runtime"
GUARD_FILE = GUARD_DIR / "active_pids.json"

_LOCK = threading.Lock()
_ACTIVE: dict[int, str] = {}  # pid -> human-readable name
_INSTALLED = False


# ---------------------------------------------------------------------------
# Persistence — write our PID set to disk so a hard crash can be recovered
# ---------------------------------------------------------------------------

def _persist_locked() -> None:
    """Write _ACTIVE to disk. Caller must hold _LOCK."""
    try:
        GUARD_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "owner_pid": os.getpid(),
            "children": {str(pid): name for pid, name in _ACTIVE.items()},
        }
        GUARD_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass  # best effort — don't crash the parent over a guard file


# ---------------------------------------------------------------------------
# Public API: register / unregister
# ---------------------------------------------------------------------------

def register_child(pid: int, name: str = "") -> None:
    """Register a child PID so it gets killed on parent exit."""
    if not isinstance(pid, int) or pid <= 0:
        return
    with _LOCK:
        _ACTIVE[pid] = name or f"pid={pid}"
        _persist_locked()


def unregister_child(pid: int) -> None:
    """Unregister a child PID (call after the child exits cleanly)."""
    with _LOCK:
        _ACTIVE.pop(pid, None)
        _persist_locked()


def list_children() -> dict[int, str]:
    """Return a copy of the current PID set (for diagnostics)."""
    with _LOCK:
        return dict(_ACTIVE)


# ---------------------------------------------------------------------------
# Tree-killing — walk the descendant tree, then kill parent
# ---------------------------------------------------------------------------

def _kill_tree(pid: int, name: str = "") -> bool:
    """Kill `pid` and all of its descendants. Returns True if anything died."""
    if _PSUTIL_OK:
        try:
            proc = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return False
        # Snapshot children BEFORE killing the parent (so the tree is intact)
        try:
            descendants = proc.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            descendants = []
        # Kill descendants leaf-first (psutil already orders correctly)
        for child in descendants:
            try:
                child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        # Kill the registered parent last
        try:
            proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        # Wait briefly so OS can reap them — avoids the "ghost" PID issue
        try:
            psutil.wait_procs([proc, *descendants], timeout=3)
        except (psutil.TimeoutExpired, Exception):  # noqa: BLE001
            pass
        return True

    # Fallback: no psutil → can only kill direct child
    try:
        if hasattr(signal, "SIGTERM"):
            os.kill(pid, signal.SIGTERM)
        else:
            os.kill(pid, 15)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def kill_all() -> int:
    """Kill every registered child + their descendants. Returns count killed."""
    with _LOCK:
        items = list(_ACTIVE.items())
        _ACTIVE.clear()
        _persist_locked()
    killed = 0
    for pid, name in items:
        if _kill_tree(pid, name):
            killed += 1
    return killed


# ---------------------------------------------------------------------------
# Startup orphan sweep — kill leftovers from a previous crashed run
# ---------------------------------------------------------------------------

def _looks_like_our_proc(proc) -> bool:  # type: ignore[no-untyped-def]
    """Heuristic: only kill processes that look like they belong to this codebase.

    We refuse to kill anything that doesn't have ``python`` in the name AND
    doesn't have an LLM_Factory-ish path or scripts/ dir in its cmdline. This
    is the safety net that prevents us from nuking unrelated python processes
    on a shared machine.
    """
    if not _PSUTIL_OK:
        return False  # can't verify → don't kill
    try:
        name = (proc.name() or "").lower()
        if "python" not in name and "py" not in name:
            return False
        cmdline = " ".join(proc.cmdline() or []).lower().replace("\\", "/")
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
    # Must reference this codebase somehow
    markers = ("llm_factory", "scripts/distill", "scripts/multi_teacher", "scripts/speed_graduation", "scripts/skill_branch")
    return any(m in cmdline for m in markers)


def sweep_orphans() -> int:
    """Read the recovery file from a previous run and kill any leftovers.

    Returns the number of orphan PIDs we killed. Refuses to touch the
    recovery file's "owner_pid" if that PID is still alive AND looks like
    a valid distill_server (so two simultaneous runs don't fight).
    """
    if not GUARD_FILE.exists():
        return 0
    try:
        data = json.loads(GUARD_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return 0
    prev_owner = data.get("owner_pid")
    children = data.get("children") or {}
    if not children:
        # Nothing to clean — refresh the file with our identity
        with _LOCK:
            _persist_locked()
        return 0

    # If the previous owner is still alive AND looks like one of ours,
    # the previous server is still running — don't kill its children.
    if prev_owner and _PSUTIL_OK:
        try:
            owner_pid = int(prev_owner)
            if owner_pid != os.getpid() and psutil.pid_exists(owner_pid):
                owner_proc = psutil.Process(owner_pid)
                if _looks_like_our_proc(owner_proc):
                    sys.stderr.write(
                        f"[process_guard] previous owner pid={owner_pid} still alive — skipping sweep\n"
                    )
                    return 0
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
            pass

    killed = 0
    for pid_str, name in children.items():
        try:
            pid = int(pid_str)
        except (TypeError, ValueError):
            continue
        if not _PSUTIL_OK:
            try:
                os.kill(pid, signal.SIGTERM)
                killed += 1
                sys.stderr.write(f"[process_guard] swept orphan pid={pid} name={name}\n")
            except (ProcessLookupError, PermissionError, OSError):
                continue
            continue
        try:
            proc = psutil.Process(pid)
        except psutil.NoSuchProcess:
            continue
        if not _looks_like_our_proc(proc):
            sys.stderr.write(
                f"[process_guard] skipping pid={pid} ({name}) — doesn't look like ours\n"
            )
            continue
        if _kill_tree(pid, name):
            killed += 1
            sys.stderr.write(
                f"[process_guard] swept orphan pid={pid} name={name}\n"
            )

    # Truncate the recovery file with our identity
    with _LOCK:
        _persist_locked()
    return killed


# ---------------------------------------------------------------------------
# Atexit + signal handlers
# ---------------------------------------------------------------------------

def _atexit_handler() -> None:
    n = kill_all()
    if n:
        sys.stderr.write(f"[process_guard] killed {n} child process tree(s) at exit\n")


def _signal_handler(signum, _frame) -> None:  # type: ignore[no-untyped-def]
    sys.stderr.write(f"[process_guard] caught signal {signum}, killing children\n")
    n = kill_all()
    if n:
        sys.stderr.write(f"[process_guard] killed {n} child process tree(s)\n")
    # Re-raise default behavior so the parent actually exits
    # Use os._exit() to skip cleanup that might re-trigger our handlers
    try:
        sys.exit(128 + int(signum))
    except SystemExit:
        os._exit(128 + int(signum))


def install() -> int:
    """Install atexit + signal handlers and sweep any leftover orphans.

    Idempotent — safe to call multiple times. Returns the number of orphan
    processes killed during the startup sweep.
    """
    global _INSTALLED  # noqa: PLW0603
    if _INSTALLED:
        return 0
    _INSTALLED = True

    # Sweep first so we don't double-count our own PID as an orphan
    swept = sweep_orphans()

    atexit.register(_atexit_handler)

    # Install signal handlers — best effort, some signals don't exist on Windows
    for sig_name in ("SIGINT", "SIGTERM", "SIGBREAK", "SIGHUP"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _signal_handler)
        except (ValueError, OSError):
            # Some signals can only be set in the main thread; that's fine
            pass

    if swept:
        sys.stderr.write(f"[process_guard] startup sweep killed {swept} orphan(s)\n")
    return swept


# ---------------------------------------------------------------------------
# CLI: `python -m scripts.process_guard sweep` to manually clean up
# ---------------------------------------------------------------------------

def _cli() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Manually sweep orphan PIDs from a previous LLM_Factory run.",
    )
    parser.add_argument("action", choices=("sweep", "list", "status"),
                        help="sweep=kill leftovers, list=show recovery file, status=psutil check")
    args = parser.parse_args()

    if args.action == "status":
        print(f"psutil available: {_PSUTIL_OK}")
        print(f"recovery file:    {GUARD_FILE}")
        print(f"file exists:      {GUARD_FILE.exists()}")
        return 0

    if args.action == "list":
        if not GUARD_FILE.exists():
            print("(no recovery file)")
            return 0
        print(GUARD_FILE.read_text(encoding="utf-8"))
        return 0

    if args.action == "sweep":
        n = sweep_orphans()
        print(f"swept {n} orphan(s)")
        return 0

    return 1


__all__ = [
    "install",
    "register_child",
    "unregister_child",
    "list_children",
    "kill_all",
    "sweep_orphans",
]


if __name__ == "__main__":
    raise SystemExit(_cli())
