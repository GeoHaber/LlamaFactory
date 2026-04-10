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

"""Skill Branch — Phase 2 of the Stem + Branches architecture.

See LLM_STATE_OF_THE_UNION_2026-04-09.md, Part 11.

Takes a merged Speed-Run model (the "trunk") and trains a small LoRA "branch"
on top of it for one specific skill, with a replay buffer of trunk training
data mixed in to prevent catastrophic forgetting.

Usage:
    # Single skill branch
    python scripts/skill_branch.py \\
        --trunk saves/trunk_v1/merged \\
        --skill translate \\
        --skill-data data/skills/translate_pairs.jsonl \\
        --replay-data saves/trunk_v1/consensus_sft.jsonl \\
        --replay-fraction 0.15 \\
        --tag trunk_v1_translate

    # Combine multiple branches via DARE-TIES
    python scripts/skill_branch.py --combine \\
        --trunk saves/trunk_v1/merged \\
        --branches saves/trunk_v1_translate,saves/trunk_v1_coding \\
        --tag trunk_v1_full

Why a "branch" not a "fork":
  - Branch LoRAs are SMALL (rank 8) — they're a delta on top of the trunk,
    not a full re-training. ~5–50 MB on disk.
  - The trunk is FROZEN (loaded as model_name_or_path, never updated). The
    branch only learns the skill delta.
  - 15 % of the trunk's training data is replayed alongside the skill data
    so the branch can't catastrophically forget reasoning while learning
    its new trick. Research from 2024–2026 (LoRA Soups, Med-MoE-LoRA,
    SuRe) all converge on "replay 10–20 % of base data" as the simplest
    effective mitigation.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import yaml  # xray: ignore[SEC-015]

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(f"[skill_branch] {msg}", flush=True)  # xray: ignore[PY-004]


# ---------------------------------------------------------------------------
# Replay buffer construction
# ---------------------------------------------------------------------------

def _build_replay_dataset(
    skill_path: Path,
    replay_path: Path | None,
    fraction: float,
    out_path: Path,
) -> tuple[int, int]:
    """Mix `fraction * n_skill` rows of replay data into the skill JSONL.

    Returns (n_skill, n_replay) so callers can log it.
    """
    if not skill_path.exists():
        raise FileNotFoundError(f"skill data not found: {skill_path}")

    skill_rows: list[str] = [
        line.strip()
        for line in skill_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    n_skill = len(skill_rows)
    if n_skill == 0:
        raise ValueError(f"no rows in skill data: {skill_path}")

    n_replay_target = int(round(n_skill * max(0.0, fraction)))
    replay_rows: list[str] = []
    if replay_path is not None and replay_path.exists() and n_replay_target > 0:
        all_replay = [
            line.strip()
            for line in replay_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if all_replay:
            random.seed(42)  # deterministic sample for reproducibility
            replay_rows = random.sample(all_replay, min(n_replay_target, len(all_replay)))

    combined = skill_rows + replay_rows
    random.seed(123)
    random.shuffle(combined)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(combined) + "\n", encoding="utf-8")
    _log(f"built {out_path}: {n_skill} skill + {len(replay_rows)} replay = {len(combined)} total")
    return n_skill, len(replay_rows)


# ---------------------------------------------------------------------------
# YAML builders -- mirror gen_distill_configs.py shape
# ---------------------------------------------------------------------------

# Default ranks per skill family, grounded in "LoRA Learns Less and Forgets
# Less" (arXiv 2405.09673). Their headline finding: for instruction FT on
# code/math, LoRA needs a HIGH rank (64-256) to close the gap with full FT.
# Our earlier default of rank=8 was under-powered for anything beyond toy
# skills. These numbers are the practical middle ground between "big enough
# to learn the skill" and "small enough not to catastrophically forget the
# trunk." Users can override with --rank.
_SKILL_RANK_DEFAULTS = {
    "translate": 32,   # translation is easier, 32 is plenty
    "translation": 32,
    "code": 64,        # code is hard per the paper; 64 is the minimum
    "coding": 64,
    "math": 64,        # same as code
    "reasoning": 32,   # trunk already knows reasoning; small delta
    "style": 16,       # style transfer is thin
    "voice": 16,
    "domain": 32,      # generic domain adaptation
    "legal": 32,
    "medical": 32,
}


def _rank_for_skill(skill: str, override: int | None) -> int:
    """Pick a sensible LoRA rank per skill, or use the user's override."""
    if override is not None and override > 0:
        return int(override)
    return _SKILL_RANK_DEFAULTS.get(skill.strip().lower(), 32)


def _branch_sft_yaml(
    trunk_path: str,
    dataset_name: str,
    tag: str,
    cpu_safe: bool,
    rank: int,
    epochs: float,
    learning_rate: float,
) -> dict:
    """Build a LoRA SFT YAML using the merged trunk as the base.

    Defaults informed by recent literature:
      - ``lora_rank``: 32 by default, 64 for code/math (arXiv 2405.09673
        "LoRA Learns Less and Forgets Less" shows low ranks undertrain code
        tasks; rank 64+ is needed to close the gap with full FT).
      - ``lora_alpha = 2 * rank``: the alpha=2r convention from the same
        paper and Unsloth's recipe. Our earlier alpha=rank (scaling 1.0)
        was under-scaled.
      - ``lora_dropout=0.05``: slight reg to prevent skill overfit on a
        small dataset. Zero is fine if your skill set has 10k+ rows.
      - ``learning_rate=1e-4``: middle of the 1e-5 — 5e-4 sweep range the
        paper recommends. Branches need more juice per step than the trunk
        because the LoRA is learning a narrower distribution.
      - ``num_train_epochs=2.0``: skill branches benefit from a bit more
        training than the trunk's single epoch. Bumped from our earlier
        default of 1.0 after re-reading the scaling-laws paper.
    """
    return {
        "model_name_or_path": trunk_path,
        "trust_remote_code": True,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        # ── LoRA sizing — tuned per skill via _rank_for_skill() ───────────
        "lora_rank": rank,
        "lora_alpha": rank * 2,  # α = 2r (paper convention)
        "lora_dropout": 0.05,
        "lora_target": "all",
        # ── Dataset / template — match trunk template for token alignment ─
        "dataset_dir": "data",
        "dataset": dataset_name,
        "template": "qwen3_5",
        "enable_thinking": True,
        "mask_history": True,
        "cutoff_len": 4096 if cpu_safe else 8192,
        "max_samples": 50000,
        "preprocessing_num_workers": 1,
        "dataloader_num_workers": 0,
        # ── Output / logging ─────────────────────────────────────────────
        "output_dir": f"saves/{tag}/lora/sft",
        "logging_steps": 5,
        "save_steps": 100,
        "plot_loss": False,
        "overwrite_output_dir": False,
        "save_only_model": False,
        "report_to": "none",
        # ── Optimizer ────────────────────────────────────────────────────
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": learning_rate,
        "num_train_epochs": epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "optim": "adamw_torch" if cpu_safe else "adamw_8bit",
        "weight_decay": 0.001,
        "bf16": not cpu_safe,
        "fp16": False,
    }


def _branch_oft_yaml(
    trunk_path: str,
    dataset_name: str,
    tag: str,
    cpu_safe: bool,
    oft_rank: int,
    oft_block_size: int,
    epochs: float,
    learning_rate: float,
) -> dict:
    """Build an OFT SFT YAML using the merged trunk as the base.

    OFT (Orthogonal Fine-Tuning, arXiv 2311.06243) constrains weight updates
    to orthogonal rotations, which provably preserves the "hyperspherical
    energy" of pre-trained representations.  Translation: it literally cannot
    distort what the trunk already knows, by construction.

    Trade-off vs LoRA:
      - **Pro**: zero catastrophic forgetting (no replay buffer needed in theory)
      - **Con**: fewer degrees of freedom → may not reach the same peak quality
        on complex skills (code, math) that need high-rank updates.

    Parameters:
      - ``oft_rank``: intrinsic dimension of the orthogonal rotation (NOT the
        same as LoRA rank — OFT rank is typically smaller: 4-16).
      - ``oft_block_size``: controls the block-diagonal structure of the
        orthogonal matrix (default 32, from PEFT default).
      - ``oft_target="all"``: apply to all linear layers (same as LoRA).

    Note: LlamaFactory supports OFT training natively (``finetuning_type: oft``)
    but its ``export`` command currently only supports LoRA adapter merging.
    OFT adapters must be used as-is at inference time (loaded via PEFT), or
    merged using a custom PEFT script.  The ``--skip-merge`` flag is implied
    when ``--adapter-type oft`` is used.
    """
    return {
        "model_name_or_path": trunk_path,
        "trust_remote_code": True,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "oft",
        # ── OFT sizing ───────────────────────────────────────────────────
        "oft_rank": oft_rank,
        "oft_block_size": oft_block_size,
        "oft_target": "all",
        "module_dropout": 0.0,
        # ── Dataset / template — match trunk template for token alignment ─
        "dataset_dir": "data",
        "dataset": dataset_name,
        "template": "qwen3_5",
        "enable_thinking": True,
        "mask_history": True,
        "cutoff_len": 4096 if cpu_safe else 8192,
        "max_samples": 50000,
        "preprocessing_num_workers": 1,
        "dataloader_num_workers": 0,
        # ── Output / logging ─────────────────────────────────────────────
        "output_dir": f"saves/{tag}/oft/sft",
        "logging_steps": 5,
        "save_steps": 100,
        "plot_loss": False,
        "overwrite_output_dir": False,
        "save_only_model": False,
        "report_to": "none",
        # ── Optimizer ────────────────────────────────────────────────────
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": learning_rate,
        "num_train_epochs": epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "optim": "adamw_torch" if cpu_safe else "adamw_8bit",
        "weight_decay": 0.001,
        "bf16": not cpu_safe,
        "fp16": False,
    }


def _branch_merge_yaml(trunk_path: str, branch_adapter: str, tag: str) -> dict:
    """Merge the branch LoRA into the trunk to produce a deployable model."""
    return {
        "model_name_or_path": trunk_path,
        "adapter_name_or_path": branch_adapter,
        "template": "qwen3_5",
        "trust_remote_code": True,
        "export_dir": f"saves/{tag}/merged",
        "export_size": 5,
        "export_device": "cpu",
        "export_legacy_format": False,
    }


def _dare_ties_yaml(trunk_path: str, branch_paths: list[str]) -> dict:
    """mergekit DARE-TIES merge of N branch merged models into the trunk."""
    weight = round(1.0 / max(len(branch_paths), 1), 3)
    return {
        "merge_method": "dare_ties",
        "base_model": trunk_path,
        "parameters": {
            "normalize": True,
            "int8_mask": True,
        },
        "models": [
            {"model": p, "parameters": {"weight": weight, "density": 0.5}}
            for p in branch_paths
        ],
        "dtype": "float16",
    }


# ---------------------------------------------------------------------------
# dataset_info.json registration
# ---------------------------------------------------------------------------

def _register_dataset(name: str, jsonl_path: Path) -> None:
    info_path = ROOT / "data" / "dataset_info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info: dict = {}
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            info = {}
    # Use repo-relative POSIX path so the YAML is portable across machines
    try:
        rel = jsonl_path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        rel = str(jsonl_path).replace("\\", "/")
    info[name] = {
        "file_name": rel,
        "columns": {"prompt": "instruction", "response": "output"},
    }
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _log(f"registered dataset '{name}' -> {rel}")


# ---------------------------------------------------------------------------
# Subprocess runner — uses process_guard if available
# ---------------------------------------------------------------------------

def _run(cmd: list[str], log_path: Path) -> int:
    _log(f"running: {' '.join(cmd)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Lazy-import process_guard so this script also works on machines that
    # don't have psutil installed (process_guard degrades gracefully).
    try:
        import process_guard  # type: ignore  # xray: ignore[SEC-015]
        process_guard.install()
    except ImportError:
        process_guard = None  # type: ignore

    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(ROOT),
        )
        if process_guard is not None:
            process_guard.register_child(proc.pid, " ".join(cmd[:2]))
        try:
            rc = proc.wait()
        finally:
            if process_guard is not None:
                process_guard.unregister_child(proc.pid)
    return rc


# ---------------------------------------------------------------------------
# Subcommand: branch
# ---------------------------------------------------------------------------

def cmd_branch(args: argparse.Namespace) -> int:
    adapter_type: str = getattr(args, "adapter_type", "lora")

    trunk = Path(args.trunk).resolve()
    if not (trunk / "config.json").exists():
        _log(f"ERROR: trunk {trunk} doesn't look like a merged HF model (no config.json)")
        return 1

    skill_data = Path(args.skill_data).resolve()
    if not skill_data.exists():
        _log(f"ERROR: skill data {skill_data} not found")
        return 1

    replay_data: Path | None = None
    if args.replay_data:
        replay_data = Path(args.replay_data).resolve()
        if not replay_data.exists():
            _log(f"WARNING: replay data {replay_data} not found — proceeding without replay")
            replay_data = None

    if adapter_type == "oft" and replay_data is None:
        _log("INFO: OFT preserves trunk representations by construction — "
             "replay buffer is optional (but still recommended for safety)")

    # Build the mixed (skill + replay) dataset
    mixed_path = ROOT / "data" / "branches" / args.tag / "training.jsonl"
    n_skill, n_replay = _build_replay_dataset(
        skill_data, replay_data, args.replay_fraction, mixed_path
    )

    # Register it under a fresh name
    dataset_name = f"{args.tag}_branch"
    _register_dataset(dataset_name, mixed_path)

    trunk_str = str(trunk).replace("\\", "/")

    # ── Rank selection ───────────────────────────────────────────────────
    rank_arg = args.rank
    if adapter_type == "lora":
        if rank_arg == "auto":
            # Empirical rank probing — run 50-step sweeps at multiple ranks
            _log("rank=auto: launching empirical rank probe ...")
            try:
                sys.path.insert(0, str(ROOT / "scripts"))
                from rank_probe import probe_rank
                rank, probe_details = probe_rank(
                    trunk_path=trunk_str,
                    dataset_name=dataset_name,
                    tag=f"probe_{args.tag}",
                    cpu_safe=args.cpu_safe,
                )
                _log(f"rank probe selected rank={rank}")
            except Exception as exc:
                _log(f"WARNING: rank probe failed ({exc}), falling back to per-skill default")
                rank = _rank_for_skill(args.skill, None)
        elif rank_arg is not None:
            rank = int(rank_arg)
        else:
            rank = _rank_for_skill(args.skill, None)
        _log(f"LoRA rank={rank} (alpha={rank * 2}) for skill={args.skill}")
    else:
        # OFT mode — use oft_rank and oft_block_size
        oft_rank = int(rank_arg) if rank_arg and rank_arg != "auto" else args.oft_rank
        _log(f"OFT rank={oft_rank} block_size={args.oft_block_size} for skill={args.skill}")

    # ── Generate training YAML ───────────────────────────────────────────
    if adapter_type == "lora":
        cfg = _branch_sft_yaml(
            trunk_path=trunk_str,
            dataset_name=dataset_name,
            tag=args.tag,
            cpu_safe=args.cpu_safe,
            rank=rank,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )
        adapter_subdir = "lora/sft"
    else:
        cfg = _branch_oft_yaml(
            trunk_path=trunk_str,
            dataset_name=dataset_name,
            tag=args.tag,
            cpu_safe=args.cpu_safe,
            oft_rank=oft_rank,
            oft_block_size=args.oft_block_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )
        adapter_subdir = "oft/sft"

    yaml_path = ROOT / "examples" / "distillation" / "auto" / f"{args.tag}_branch_sft.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with yaml_path.open("w", encoding="utf-8") as f:
        f.write(f"### Auto-generated branch SFT config (skill={args.skill}, adapter={adapter_type})\n\n")
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    _log(f"wrote {yaml_path}")

    if args.dry_run:
        _log("--dry-run: stopping before training")
        return 0

    # ── Train ────────────────────────────────────────────────────────────
    log_dir = ROOT / "saves" / args.tag / "logs"
    rc = _run(["llamafactory-cli", "train", str(yaml_path)], log_dir / "branch_sft.log")
    if rc != 0:
        _log(f"ERROR: branch SFT failed (rc={rc}). See {log_dir / 'branch_sft.log'}")
        return rc

    # ── Merge (LoRA only — OFT export not supported by llamafactory-cli) ─
    if adapter_type == "oft":
        _log(f"OFT adapter ready at saves/{args.tag}/{adapter_subdir}/")
        _log("NOTE: OFT merge via 'llamafactory-cli export' is not supported.")
        _log("      Use the adapter directly with PEFT at inference time, or")
        _log("      merge manually with: model.merge_and_unload()")
        _log(f"  - training data:   {mixed_path}")
        _log(f"  - replay fraction: {args.replay_fraction} ({n_skill} skill + {n_replay} replay)")
        return 0

    if args.skip_merge:
        _log("--skip-merge: stopping after SFT")
        _log(f"branch LoRA ready at saves/{args.tag}/{adapter_subdir}/")
        return 0

    branch_adapter = f"saves/{args.tag}/{adapter_subdir}"
    if not (ROOT / branch_adapter).exists():
        _log(f"ERROR: expected adapter at {branch_adapter} but it's missing")
        return 1

    merge_cfg = _branch_merge_yaml(
        trunk_path=trunk_str,
        branch_adapter=branch_adapter,
        tag=args.tag,
    )
    merge_yaml_path = ROOT / "examples" / "distillation" / "auto" / f"{args.tag}_branch_merge.yaml"
    with merge_yaml_path.open("w", encoding="utf-8") as f:
        f.write(f"### Auto-generated branch merge config (skill={args.skill})\n\n")
        yaml.dump(merge_cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    _log(f"wrote {merge_yaml_path}")

    rc = _run(["llamafactory-cli", "export", str(merge_yaml_path)], log_dir / "branch_merge.log")
    if rc != 0:
        _log(f"WARNING: branch merge failed (rc={rc}). The LoRA at {branch_adapter} is still usable.")
        return rc

    _log(f"branch ready: saves/{args.tag}/")
    _log(f"  - adapter ({adapter_type}): {branch_adapter}")
    _log(f"  - merged model:    saves/{args.tag}/merged/")
    _log(f"  - training data:   {mixed_path}")
    _log(f"  - replay fraction: {args.replay_fraction} ({n_skill} skill + {n_replay} replay)")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: combine (DARE-TIES merge of N branches)
# ---------------------------------------------------------------------------

def cmd_combine(args: argparse.Namespace) -> int:
    trunk = Path(args.trunk).resolve()
    if not (trunk / "config.json").exists():
        _log(f"ERROR: trunk {trunk} doesn't look like a merged HF model")
        return 1

    branch_dirs = [b.strip() for b in args.branches.split(",") if b.strip()]
    if len(branch_dirs) < 2:
        _log("ERROR: --combine needs at least 2 branches in --branches")
        return 1

    # Resolve each branch — accept either the saves/<tag>/ root or saves/<tag>/merged/
    resolved: list[str] = []
    for b in branch_dirs:
        bp = Path(b).resolve()
        if (bp / "merged" / "config.json").exists():
            bp = bp / "merged"
        elif not (bp / "config.json").exists():
            _log(f"ERROR: branch {bp} has no merged model (no config.json)")
            return 1
        resolved.append(str(bp).replace("\\", "/"))

    cfg = _dare_ties_yaml(str(trunk).replace("\\", "/"), resolved)
    yaml_path = ROOT / "examples" / "distillation" / "auto" / f"{args.tag}_combine.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with yaml_path.open("w", encoding="utf-8") as f:
        f.write("### Auto-generated DARE-TIES combine config\n\n")
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    _log(f"wrote {yaml_path}")
    _log("next step: run mergekit to actually combine the branches")
    _log(f"  mergekit-yaml {yaml_path} saves/{args.tag}/merged")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Skill Branch — Phase 2 of the Stem + Branches architecture.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # LoRA branch with auto rank selection (empirical probe)
  %(prog)s --trunk saves/trunk_v1/merged \\
           --skill translate \\
           --skill-data data/skills/translate.jsonl \\
           --replay-data saves/trunk_v1/consensus_sft.jsonl \\
           --rank auto --tag trunk_v1_translate

  # OFT branch (orthogonal FT — preserves trunk, no replay needed)
  %(prog)s --trunk saves/trunk_v1/merged \\
           --skill code \\
           --skill-data data/skills/code.jsonl \\
           --adapter-type oft --oft-rank 8 \\
           --tag trunk_v1_code_oft

  # Combine 2 branches into one model via DARE-TIES
  %(prog)s --combine \\
           --trunk saves/trunk_v1/merged \\
           --branches saves/trunk_v1_translate,saves/trunk_v1_coding \\
           --tag trunk_v1_full
""",
    )
    parser.add_argument("--trunk", help="Path to merged Speed-Run trunk model (HF dir with config.json)")
    parser.add_argument("--skill", default="skill", help="Skill name (cosmetic label)")
    parser.add_argument("--skill-data", help="Path to skill JSONL (instruction/output rows)")
    parser.add_argument("--replay-data", default="", help="Path to trunk consensus_sft.jsonl for replay buffer")
    parser.add_argument("--replay-fraction", type=float, default=0.15,
                        help="Fraction of skill-data size to mix in as replay (default 0.15). "
                             "NOT optional for LoRA continual-learning (research shows LoRA "
                             "does NOT automatically mitigate catastrophic forgetting). "
                             "10-20%% is the documented effective range. "
                             "OFT preserves trunk representations by construction — replay "
                             "is optional but recommended.")
    # ── Adapter type ─────────────────────────────────────────────────────
    parser.add_argument("--adapter-type", choices=["lora", "oft"], default="lora",
                        help="Adapter method: 'lora' (default) or 'oft' (orthogonal FT, "
                             "arXiv 2311.06243). OFT preserves trunk representations by "
                             "construction but has fewer degrees of freedom.")
    # ── Rank (LoRA) ──────────────────────────────────────────────────────
    parser.add_argument("--rank", default=None,
                        help="LoRA rank. 'auto' = empirical probe (runs 50-step sweeps "
                             "at ranks [16,32,64,128] and picks the smallest that learns "
                             "effectively). Integer = override. Omit = per-skill default "
                             "(translate=32, code=64, math=64, style=16). "
                             "Grounded in arXiv 2405.09673.")
    # ── OFT-specific ─────────────────────────────────────────────────────
    parser.add_argument("--oft-rank", type=int, default=8,
                        help="OFT intrinsic dimension (default 8). NOT the same as LoRA "
                             "rank — OFT rank is typically much smaller (4-16). Only used "
                             "when --adapter-type oft.")
    parser.add_argument("--oft-block-size", type=int, default=32,
                        help="OFT block size (default 32, PEFT default). Controls the "
                             "block-diagonal structure of the orthogonal matrix.")
    # ── Training ─────────────────────────────────────────────────────────
    parser.add_argument("--epochs", type=float, default=2.0,
                        help="Training epochs (default 2.0, bumped from 1.0 per paper guidance).")
    parser.add_argument("--learning-rate", type=float, default=1.0e-4,
                        help="Learning rate (default 1e-4, middle of the 1e-5 to 5e-4 sweep "
                             "range from the paper).")
    parser.add_argument("--tag", required=True, help="Output tag (saves/<tag>/)")
    parser.add_argument("--cpu-safe", action="store_true",
                        help="Use CPU-friendly defaults (no bf16, cutoff 4096, adamw_torch)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate configs but skip training/merging")
    parser.add_argument("--skip-merge", action="store_true",
                        help="Train the LoRA but don't merge it into the trunk")
    parser.add_argument("--combine", action="store_true",
                        help="Combine mode: merge multiple branches into the trunk via DARE-TIES")
    parser.add_argument("--branches", default="",
                        help="Combine mode: comma-separated branch saves dirs")

    args = parser.parse_args()

    if args.combine:
        if not args.trunk or not args.branches:
            parser.error("--combine requires --trunk and --branches")
        return cmd_combine(args)

    if not args.trunk or not args.skill_data:
        parser.error("--trunk and --skill-data are required (or use --combine)")
    return cmd_branch(args)


if __name__ == "__main__":
    raise SystemExit(main())
