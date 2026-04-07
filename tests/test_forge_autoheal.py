"""Tests for ForgeState auto-heal (crash recovery, state persistence)."""

import json
import sys
import time
from pathlib import Path

import pytest

# Add scripts/ to path so we can import from run_student_forge
_scripts = str(Path(__file__).resolve().parent.parent / "scripts")
if _scripts not in sys.path:
    sys.path.insert(0, _scripts)

from run_student_forge import ForgeState, _find_latest_checkpoint


class TestForgeState:
    def test_blank_state(self, tmp_path):
        state = ForgeState("test", saves_dir=tmp_path)
        assert state.completed_ids() == set()
        assert state.completed_results() == []

    def test_record_and_resume(self, tmp_path):
        state = ForgeState("test", saves_dir=tmp_path)
        state.record_complete("A", {"sft_final_loss": 1.5, "elapsed_sec": 60})

        # Simulate restart — new ForgeState loads from disk
        state2 = ForgeState("test", saves_dir=tmp_path)
        assert state2.is_completed("A")
        assert not state2.is_completed("B")
        assert state2.completed_ids() == {"A"}

    def test_completed_results_reconstructed(self, tmp_path):
        state = ForgeState("test", saves_dir=tmp_path)
        state.record_complete("A", {"sft_final_loss": 1.5, "dpo_final_loss": 0.8, "elapsed_sec": 120})

        state2 = ForgeState("test", saves_dir=tmp_path)
        results = state2.completed_results()
        assert len(results) == 1
        assert results[0]["variant_id"] == "A"
        assert results[0]["ok"] is True
        assert results[0]["sft_final_loss"] == 1.5
        assert results[0]["resumed"] is True

    def test_record_failure(self, tmp_path):
        state = ForgeState("test", saves_dir=tmp_path)
        state.record_failure("B", "OOM crash", "Reduce batch size")

        state2 = ForgeState("test", saves_dir=tmp_path)
        data = json.loads((tmp_path / "test" / "forge_state.json").read_text("utf-8"))
        assert "B" in data["failed"]
        assert "OOM crash" in data["failed"]["B"]["error"]
        assert "Reduce batch size" in data["failed"]["B"]["diagnosis"]

    def test_record_finished(self, tmp_path):
        state = ForgeState("test", saves_dir=tmp_path)
        state.record_complete("A", {"sft_final_loss": 1.0})
        state.record_finished()

        data = json.loads((tmp_path / "test" / "forge_state.json").read_text("utf-8"))
        assert data["status"] == "finished"
        assert "finished_at" in data

    def test_heartbeat(self, tmp_path):
        state = ForgeState("test", saves_dir=tmp_path)
        assert state.last_heartbeat_age() is None

        state.write_heartbeat()
        age = state.last_heartbeat_age()
        assert age is not None
        assert age < 2.0  # should be near-instant

    def test_multiple_variants(self, tmp_path):
        state = ForgeState("test", saves_dir=tmp_path)
        state.record_complete("A", {"sft_final_loss": 1.5})
        state.record_complete("B", {"sft_final_loss": 2.0})
        state.record_failure("C", "timeout")

        state2 = ForgeState("test", saves_dir=tmp_path)
        assert state2.completed_ids() == {"A", "B"}
        assert len(state2.completed_results()) == 2

    def test_atomic_write_survives_reload(self, tmp_path):
        """State file should be valid JSON even after multiple rapid writes."""
        state = ForgeState("test", saves_dir=tmp_path)
        for i in range(20):
            state.record_complete(f"v{i}", {"sft_final_loss": float(i)})

        state2 = ForgeState("test", saves_dir=tmp_path)
        assert len(state2.completed_ids()) == 20


class TestFindLatestCheckpoint:
    def test_no_dir(self, tmp_path):
        assert _find_latest_checkpoint(str(tmp_path / "nonexistent")) is None

    def test_empty_dir(self, tmp_path):
        assert _find_latest_checkpoint(str(tmp_path)) is None

    def test_finds_highest(self, tmp_path):
        (tmp_path / "checkpoint-100").mkdir()
        (tmp_path / "checkpoint-200").mkdir()
        (tmp_path / "checkpoint-50").mkdir()
        result = _find_latest_checkpoint(str(tmp_path))
        assert result is not None
        assert "checkpoint-200" in result

    def test_ignores_non_checkpoint_dirs(self, tmp_path):
        (tmp_path / "logs").mkdir()
        (tmp_path / "checkpoint-100").mkdir()
        result = _find_latest_checkpoint(str(tmp_path))
        assert "checkpoint-100" in result
