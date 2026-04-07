"""End-to-end toy test — exercises purification → validation → config gen → preflight.

Uses synthetic data (5 prompts, 3 teachers) to verify the full pipeline
chain works without real models or GPU. Runs on CPU in under 30 seconds.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
_PY = sys.executable


def _make_teacher_responses(out_path: Path) -> None:
    """Generate synthetic teacher_responses.jsonl with 5 prompts × 3 teachers."""
    samples = [
        # GOLD: all agree on answer + reasoning
        {
            "id": "toy-001",
            "prompt": "What is 2+2?",
            "prompt_category": "math",
            "teachers": {
                "alpha": {"answer": "4", "raw": "The answer is 4.", "thought": "Two plus two equals four by basic addition."},
                "beta": {"answer": "4", "raw": "It equals 4.", "thought": "Simple addition gives us four from two plus two."},
                "gamma": {"answer": "4", "raw": "2+2=4", "thought": "Basic arithmetic: two added to two makes four."},
            },
            "routed_to": ["alpha", "beta", "gamma"],
        },
        # GOLD: all agree
        {
            "id": "toy-002",
            "prompt": "What color is the sky?",
            "prompt_category": "science",
            "teachers": {
                "alpha": {"answer": "blue", "raw": "The sky is blue.", "thought": "Rayleigh scattering makes sky blue."},
                "beta": {"answer": "blue", "raw": "Blue.", "thought": "Scattering of light makes it blue."},
                "gamma": {"answer": "blue", "raw": "The sky appears blue.", "thought": "Light scattering causes blue appearance."},
            },
            "routed_to": ["alpha", "beta", "gamma"],
        },
        # SILVER: majority agree on answer, reasoning differs
        {
            "id": "toy-003",
            "prompt": "Translate 'hello' to French.",
            "prompt_category": "translation",
            "teachers": {
                "alpha": {"answer": "bonjour", "raw": "Bonjour.", "thought": "Hello in French is bonjour, a formal greeting."},
                "beta": {"answer": "bonjour", "raw": "The translation is bonjour.", "thought": "Completely unrelated reasoning about German language history."},
                "gamma": {"answer": "salut", "raw": "Salut!", "thought": "Informal hello in French is salut."},
            },
            "routed_to": ["alpha", "beta", "gamma"],
        },
        # DROP: all disagree
        {
            "id": "toy-004",
            "prompt": "What is the best programming language?",
            "prompt_category": "opinion",
            "teachers": {
                "alpha": {"answer": "Python", "raw": "Python is the best.", "thought": "Easy to learn."},
                "beta": {"answer": "Rust", "raw": "Rust is the best.", "thought": "Memory safety."},
                "gamma": {"answer": "Haskell", "raw": "Haskell is the best.", "thought": "Type system."},
            },
            "routed_to": ["alpha", "beta", "gamma"],
        },
        # GOLD: single teacher
        {
            "id": "toy-005",
            "prompt": "Simplify: 6/3",
            "prompt_category": "math",
            "teachers": {
                "alpha": {"answer": "2", "raw": "6 divided by 3 is 2."},
            },
            "routed_to": ["alpha"],
        },
    ]
    with out_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


class TestEndToEndToy:
    """Full pipeline test with synthetic data, no GPU needed."""

    @pytest.fixture(autouse=True)
    def setup_tmpdir(self, tmp_path):
        self.tmp = tmp_path
        self.responses_path = tmp_path / "teacher_responses.jsonl"
        self.purified_dir = tmp_path / "purified"
        _make_teacher_responses(self.responses_path)

    def test_step1_purification(self):
        """Purify teacher responses into GOLD/SILVER/DROP tiers."""
        proc = subprocess.run(
            [_PY, str(_SCRIPTS_DIR / "purify_teacher_outputs.py"),
             "--input", str(self.responses_path),
             "--out-dir", str(self.purified_dir)],
            capture_output=True, text=True, timeout=60,
        )
        assert proc.returncode == 0, f"Purification failed: {proc.stderr}"

        # Check outputs exist
        gold_path = self.purified_dir / "consensus_sft.jsonl"
        silver_path = self.purified_dir / "conflict_dpo.jsonl"
        drop_path = self.purified_dir / "dropped_log.jsonl"
        report_path = self.purified_dir / "purification_report.json"

        assert gold_path.exists(), "consensus_sft.jsonl not created"
        assert silver_path.exists(), "conflict_dpo.jsonl not created"
        assert drop_path.exists(), "dropped_log.jsonl not created"
        assert report_path.exists(), "purification_report.json not created"

        # Verify report
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["total_samples"] == 5
        assert report["gold_count"] >= 2  # At least toy-001, toy-002, toy-005
        assert report["dropped_count"] >= 1  # At least toy-004

    def test_step2_purification_resume(self):
        """Resume purification skips already-processed samples."""
        # Run once
        subprocess.run(
            [_PY, str(_SCRIPTS_DIR / "purify_teacher_outputs.py"),
             "--input", str(self.responses_path),
             "--out-dir", str(self.purified_dir)],
            capture_output=True, text=True, timeout=60,
        )
        # Run again with --resume
        proc = subprocess.run(
            [_PY, str(_SCRIPTS_DIR / "purify_teacher_outputs.py"),
             "--input", str(self.responses_path),
             "--out-dir", str(self.purified_dir),
             "--resume"],
            capture_output=True, text=True, timeout=60,
        )
        assert proc.returncode == 0
        assert "already processed" in proc.stdout.lower() or "skipping" in proc.stdout.lower()

    def test_step3_validation(self):
        """Validate purified datasets."""
        # First purify
        subprocess.run(
            [_PY, str(_SCRIPTS_DIR / "purify_teacher_outputs.py"),
             "--input", str(self.responses_path),
             "--out-dir", str(self.purified_dir)],
            capture_output=True, text=True, timeout=60,
        )

        proc = subprocess.run(
            [_PY, str(_SCRIPTS_DIR / "validate_datasets.py"),
             "--sft-data", str(self.purified_dir / "consensus_sft.jsonl"),
             "--dpo-data", str(self.purified_dir / "conflict_dpo.jsonl"),
             "--no-ds-info"],
            capture_output=True, text=True, timeout=60,
        )
        assert proc.returncode == 0, f"Validation failed: {proc.stderr}"
        assert "Dataset Validation Report" in proc.stdout

    def test_step4_teacher_profile(self):
        """Generate teacher quality profile."""
        # First purify
        subprocess.run(
            [_PY, str(_SCRIPTS_DIR / "purify_teacher_outputs.py"),
             "--input", str(self.responses_path),
             "--out-dir", str(self.purified_dir)],
            capture_output=True, text=True, timeout=60,
        )

        json_out = self.tmp / "profile.json"
        proc = subprocess.run(
            [_PY, str(_SCRIPTS_DIR / "teacher_profile.py"),
             "--responses", str(self.responses_path),
             "--purified-dir", str(self.purified_dir),
             "--json-out", str(json_out)],
            capture_output=True, text=True, timeout=60,
        )
        assert proc.returncode == 0, f"Teacher profile failed: {proc.stderr}"
        assert json_out.exists()

        profile = json.loads(json_out.read_text(encoding="utf-8"))
        assert profile["teacher_count"] == 3  # alpha, beta, gamma
        assert profile["total_prompts"] == 5

    def test_step5_prompt_difficulty(self):
        """Score prompt difficulty from teacher responses."""
        scored_out = self.tmp / "scored.jsonl"
        proc = subprocess.run(
            [_PY, str(_SCRIPTS_DIR / "prompt_difficulty.py"),
             "--teacher-responses", str(self.responses_path),
             "--out-scored", str(scored_out),
             "--histogram"],
            capture_output=True, text=True, timeout=60,
        )
        assert proc.returncode == 0, f"Difficulty scoring failed: {proc.stderr}"
        assert scored_out.exists()

        scored = []
        for line in scored_out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                scored.append(json.loads(line))
        assert len(scored) == 5

    def test_step6_pipeline_preflight(self):
        """Run preflight checks on a synthetic prompts file."""
        proc = subprocess.run(
            [_PY, str(_SCRIPTS_DIR / "pipeline_preflight.py"),
             "--prompts", str(self.responses_path),
             "--no-deps", "--no-disk"],
            capture_output=True, text=True, timeout=60,
        )
        assert proc.returncode == 0, f"Preflight failed: {proc.stderr}"
        assert "Pipeline Preflight Report" in proc.stdout

    def test_step7_pipeline_events(self):
        """Verify pipeline event logger works."""
        sys.path.insert(0, str(_SCRIPTS_DIR))
        from pipeline_events import PipelineLogger, load_events, summarize_events

        log_path = self.tmp / "test_events.jsonl"
        with PipelineLogger("test", str(log_path), echo=False) as log:
            log.info("started", count=5)
            log.warn("low_data", threshold=10)
            log.info("finished")

        events = load_events(log_path)
        assert len(events) == 3
        assert events[0]["event"] == "started"
        assert events[0]["stage"] == "test"
        assert events[1]["level"] == "WARN"

        summary = summarize_events(events)
        assert summary["test"]["INFO"] == 2
        assert summary["test"]["WARN"] == 1

    def test_full_chain(self):
        """Run purification → validation → teacher profile in sequence."""
        # 1. Purify
        r1 = subprocess.run(
            [_PY, str(_SCRIPTS_DIR / "purify_teacher_outputs.py"),
             "--input", str(self.responses_path),
             "--out-dir", str(self.purified_dir)],
            capture_output=True, text=True, timeout=60,
        )
        assert r1.returncode == 0, f"Purify failed: {r1.stderr}"

        # 2. Validate
        r2 = subprocess.run(
            [_PY, str(_SCRIPTS_DIR / "validate_datasets.py"),
             "--sft-data", str(self.purified_dir / "consensus_sft.jsonl"),
             "--dpo-data", str(self.purified_dir / "conflict_dpo.jsonl"),
             "--no-ds-info",
             "--json-out", str(self.tmp / "validation.json")],
            capture_output=True, text=True, timeout=60,
        )
        assert r2.returncode == 0, f"Validate failed: {r2.stderr}"

        # 3. Teacher profile
        r3 = subprocess.run(
            [_PY, str(_SCRIPTS_DIR / "teacher_profile.py"),
             "--responses", str(self.responses_path),
             "--purified-dir", str(self.purified_dir),
             "--json-out", str(self.tmp / "profile.json")],
            capture_output=True, text=True, timeout=60,
        )
        assert r3.returncode == 0, f"Profile failed: {r3.stderr}"

        # Verify all outputs exist
        assert (self.purified_dir / "purification_report.json").exists()
        assert (self.tmp / "validation.json").exists()
        assert (self.tmp / "profile.json").exists()

        # Check validation passed
        val = json.loads((self.tmp / "validation.json").read_text(encoding="utf-8"))
        assert val["passed"]
