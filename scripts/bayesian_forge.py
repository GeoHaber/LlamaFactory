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

"""Bayesian Hyperparameter Search for Student Forge.

Uses Optuna TPE sampler to optimise LoRA rank, learning rate, and epoch count
across Forge Matrix variants. Each Optuna trial runs one complete training variant
and returns the final SFT loss as the objective.

Iteration 1 additions:
* ``--pruner`` chooses an Optuna pruner (``none`` | ``hyperband`` | ``median``
  | ``successive_halving``). Hyperband cuts total trial-hours by ~3x compared
  to running every trial to completion under TPE alone.
* ``--enable-liger`` / ``--use-muon`` propagate the Liger Kernel and Muon
  optimizer flags into every per-trial YAML, matching the
  ``examples/distillation/*_fast.yaml`` profiles.
* Trials now stream intermediate losses from ``trainer_log.jsonl`` while the
  subprocess is alive, calling ``trial.report()`` and ``trial.should_prune()``
  so the pruner can actually kill bad trials early.

Usage:
    python scripts/bayesian_forge.py \
        --base-matrix data/forge_matrix/zena007_matrix.yaml \
        --tag zena007_bayes \
        --n-trials 20 \
        --pruner hyperband \
        --enable-liger \
        --use-muon \
        --study-name zena007_hyperparam \
        --py .venv-py314/Scripts/python.exe

Requires: pip install optuna  (optional dependency)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path


try:
    import optuna
except ImportError:
    optuna = None  # type: ignore[assignment]

try:
    import yaml
except ImportError:
    print("PyYAML required: pip install pyyaml", file=sys.stderr)
    raise SystemExit(1)


def _parse_trainer_log_entry(entry: dict) -> tuple[int | None, float | None, float | None]:
    """Extract (step, loss, eval_loss) from one trainer_log.jsonl entry.

    Returns (None, None, None) for entries that contain neither loss nor eval_loss.
    """
    if not isinstance(entry, dict):
        return None, None, None
    step = entry.get("current_steps") or entry.get("step")
    loss = entry.get("loss")
    eval_loss = entry.get("eval_loss")
    if loss is None and eval_loss is None:
        return None, None, None
    try:
        step_i = int(step) if step is not None else None
    except (TypeError, ValueError):
        step_i = None
    return step_i, loss, eval_loss


def _run_single_trial(
    trial_id: str,
    model: str,
    lora_rank: int,
    learning_rate: float,
    num_epochs: int,
    tag: str,
    sft_dataset: str,
    template: str,
    cpu_safe: bool,
    py: str,
    eval_probes: str = "",
    enable_liger: bool = False,
    use_muon: bool = False,
    intermediate_callback: Callable[[int, float], bool] | None = None,
    poll_interval: float = 5.0,
) -> float | None:
    """Run a single SFT training variant and return objective metric.

    If ``eval_probes`` is provided, returns validation loss (``eval_loss``) to
    avoid overfitting. Otherwise falls back to final training loss.

    If ``intermediate_callback`` is provided, this function polls the
    ``trainer_log.jsonl`` file once per ``poll_interval`` seconds while the
    training subprocess is alive. For each new (step, loss) pair the callback
    is invoked. If the callback returns ``True``, the subprocess is killed and
    the function returns ``None`` (caller should treat as
    ``optuna.TrialPruned``).
    """
    import subprocess

    output_dir = f"saves/{tag}/bayesian/{trial_id}/lora/sft"
    cfg: dict = {
        "model_name_or_path": model,
        "trust_remote_code": True,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": lora_rank,
        "lora_target": "all",
        "dataset_dir": "data",
        "dataset": sft_dataset,
        "template": template,
        "cutoff_len": 1024,
        "max_samples": 10000,
        "preprocessing_num_workers": 1,
        "dataloader_num_workers": 0,
        "output_dir": output_dir,
        "logging_steps": 5,
        "save_steps": 500,
        "plot_loss": False,
        "overwrite_output_dir": True,
        "save_only_model": True,
        "report_to": "none",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": learning_rate,
        "num_train_epochs": num_epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "bf16": not cpu_safe,
        "fp16": False,
    }

    # Iter 1: optional Liger Kernel + Muon optimizer (require CUDA + liger-kernel pkg).
    if enable_liger:
        cfg["enable_liger_kernel"] = True
    if use_muon:
        cfg["use_muon"] = True

    # If eval probes provided, add eval config for validation loss
    if eval_probes:
        cfg["eval_dataset"] = eval_probes
        cfg["eval_steps"] = 50
        cfg["per_device_eval_batch_size"] = 1

    yaml_path = Path(f"saves/{tag}/bayesian/{trial_id}/config.yaml")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    # Redirect subprocess output to a per-trial log file so the parent doesn't
    # have to drain a PIPE (which can deadlock with long runs).
    trial_log_path = yaml_path.parent / "trial.log"
    trial_log_fh = trial_log_path.open("wb")

    log_path = Path(output_dir) / "trainer_log.jsonl"
    last_loss: float | None = None
    last_eval_loss: float | None = None

    try:
        proc = subprocess.Popen(
            [py, "-m", "llamafactory.cli", "train", str(yaml_path)],
            stdout=trial_log_fh,
            stderr=subprocess.STDOUT,
        )

        seen_lines = 0
        pruned = False

        while proc.poll() is None:
            time.sleep(poll_interval)
            if intermediate_callback is None or not log_path.exists():
                continue
            try:
                lines = log_path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            new_lines = lines[seen_lines:]
            seen_lines = len(lines)
            for raw in new_lines:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    continue
                step, loss, eval_loss = _parse_trainer_log_entry(entry)
                if eval_loss is not None:
                    last_eval_loss = eval_loss
                if loss is not None:
                    last_loss = loss
                if step is None:
                    continue
                report_value = eval_loss if eval_loss is not None else loss
                if report_value is None:
                    continue
                if intermediate_callback(step, float(report_value)):
                    pruned = True
                    break
            if pruned:
                break

        if pruned:
            try:
                proc.kill()
            except OSError:
                pass
            proc.wait()
            return None

        proc.wait()
        if proc.returncode != 0:
            return None
    finally:
        trial_log_fh.close()

    # Final read: pick up any tail lines flushed after the last poll tick.
    if not log_path.exists():
        return last_eval_loss if last_eval_loss is not None else last_loss

    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        _, loss, eval_loss = _parse_trainer_log_entry(entry)
        if eval_loss is not None:
            last_eval_loss = eval_loss
        if loss is not None:
            last_loss = loss

    # Return eval_loss if available (better generalization signal), else training loss
    return last_eval_loss if last_eval_loss is not None else last_loss


def main() -> int:
    if optuna is None:
        print("ERROR: optuna not installed. Run: pip install optuna", file=sys.stderr)
        return 1

    parser = argparse.ArgumentParser(
        description="Bayesian hyperparameter search for Student Forge.",
        epilog="""\
examples:
  %(prog)s --base-matrix data/forge_matrix/zena007_matrix.yaml --n-trials 20
  %(prog)s --base-matrix data/forge_matrix/zena007_matrix.yaml --n-trials 50 --lr-min 1e-5 --lr-max 5e-4
  %(prog)s --base-matrix data/forge_matrix/zena007_matrix.yaml --rank-choices 8,16,32 --epoch-max 10
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-matrix", required=True, help="Base forge matrix YAML (used for model/template/data).")
    parser.add_argument("--tag", default="bayes", help="Run tag for saves directory.")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials.")
    parser.add_argument("--study-name", default="forge_hyperparam", help="Optuna study name.")
    parser.add_argument("--py", default=".venv-py314/Scripts/python.exe", help="Python interpreter.")
    parser.add_argument("--lr-min", type=float, default=1e-6, help="Min learning rate.")
    parser.add_argument("--lr-max", type=float, default=1e-4, help="Max learning rate.")
    parser.add_argument("--rank-choices", type=str, default="8,16,32,64", help="Comma-separated LoRA rank choices.")
    parser.add_argument("--epoch-min", type=int, default=1, help="Min epochs.")
    parser.add_argument("--epoch-max", type=int, default=5, help="Max epochs.")
    parser.add_argument("--timeout", type=int, default=0, help="Total search timeout in seconds (0=unlimited).")
    parser.add_argument("--eval-probes", default="", help="Eval probes dataset name for validation loss (prevents overfitting).")
    parser.add_argument(
        "--pruner",
        choices=["none", "hyperband", "median", "successive_halving"],
        default="none",
        help="Optuna pruner: 'hyperband' / 'successive_halving' give ~3x trial-hour savings.",
    )
    parser.add_argument(
        "--pruner-min-resource",
        type=int,
        default=100,
        help="Min training steps a trial must run before becoming prunable (Hyperband / SH only).",
    )
    parser.add_argument(
        "--pruner-max-resource",
        type=int,
        default=2000,
        help="Max training steps the pruner will plan for (Hyperband only).",
    )
    parser.add_argument(
        "--pruner-reduction-factor",
        type=int,
        default=3,
        help="Reduction factor for Hyperband / Successive Halving (default 3).",
    )
    parser.add_argument(
        "--enable-liger",
        action="store_true",
        help="Enable Liger Kernel (~+20%% throughput / -60%% peak VRAM) in every trial. "
             "Requires CUDA + `pip install liger-kernel`.",
    )
    parser.add_argument(
        "--use-muon",
        action="store_true",
        help="Use the Muon optimizer for 2D params in every trial (~52%% AdamW FLOPs).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds between trainer_log.jsonl polls for intermediate pruning (default 5).",
    )
    args = parser.parse_args()

    matrix = yaml.safe_load(Path(args.base_matrix).read_text(encoding="utf-8"))
    # Use first variant's model as default
    first_variant = next(iter(matrix.get("variants", {}).values()), {})
    model = first_variant.get("model", "")
    template = matrix.get("template", "qwen")
    cpu_safe = matrix.get("cpu_safe", True)
    sft_data = matrix.get("sft_data", "")
    tag = args.tag

    # Derive SFT dataset name from tag
    sft_ds_name = f"{tag}_forge_train_sft"

    rank_choices = [int(r.strip()) for r in args.rank_choices.split(",")]

    # Build the requested pruner. Hyperband is the recommended default for
    # multi-fidelity HPO and is ~3x faster than running every trial to
    # completion under TPE alone.
    pruner: optuna.pruners.BasePruner
    if args.pruner == "hyperband":
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=args.pruner_min_resource,
            max_resource=args.pruner_max_resource,
            reduction_factor=args.pruner_reduction_factor,
        )
    elif args.pruner == "successive_halving":
        pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=args.pruner_min_resource,
            reduction_factor=args.pruner_reduction_factor,
        )
    elif args.pruner == "median":
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=args.pruner_min_resource)
    else:
        pruner = optuna.pruners.NopPruner()

    print("=== Bayesian Hyperparameter Search ===")
    print(f"Model:        {model}")
    print(f"Trials:       {args.n_trials}")
    print(f"LR range:     [{args.lr_min}, {args.lr_max}]")
    print(f"Rank opts:    {rank_choices}")
    print(f"Epoch range:  [{args.epoch_min}, {args.epoch_max}]")
    print(f"Pruner:       {args.pruner}")
    if args.enable_liger:
        print("Liger Kernel: ENABLED (requires CUDA + liger-kernel)")
    if args.use_muon:
        print("Muon:         ENABLED (2D params only; AdamW for 1D/embed/head)")
    print()

    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=pruner,
    )

    def objective(trial: optuna.Trial) -> float:
        lora_rank = trial.suggest_categorical("lora_rank", rank_choices)
        lr = trial.suggest_float("learning_rate", args.lr_min, args.lr_max, log=True)
        epochs = trial.suggest_int("num_train_epochs", args.epoch_min, args.epoch_max)

        trial_id = f"trial_{trial.number:03d}"
        print(f"\n--- Trial {trial.number}: rank={lora_rank}, lr={lr:.2e}, epochs={epochs} ---")

        def report_and_check(step: int, loss_value: float) -> bool:
            """Report intermediate loss; return True if Optuna says prune."""
            try:
                trial.report(loss_value, step)
            except Exception:  # noqa: BLE001 - Optuna can raise various pruner errors
                return False
            if trial.should_prune():
                print(f"  Trial {trial.number} pruned at step {step} (loss={loss_value:.4f})")
                return True
            return False

        t0 = time.time()
        loss = _run_single_trial(
            trial_id=trial_id,
            model=model,
            lora_rank=lora_rank,
            learning_rate=lr,
            num_epochs=epochs,
            tag=tag,
            sft_dataset=sft_ds_name,
            template=template,
            cpu_safe=cpu_safe,
            py=args.py,
            eval_probes=args.eval_probes,
            enable_liger=args.enable_liger,
            use_muon=args.use_muon,
            intermediate_callback=report_and_check if args.pruner != "none" else None,
            poll_interval=args.poll_interval,
        )
        elapsed = time.time() - t0

        if loss is None:
            # Either a hard failure or the pruner killed the subprocess; either
            # way Optuna treats this as a pruned trial.
            print(f"  Trial {trial.number} pruned/failed ({elapsed:.0f}s)")
            raise optuna.TrialPruned()

        print(f"  Trial {trial.number} => loss={loss:.4f} ({elapsed:.0f}s)")
        return loss

    timeout = args.timeout if args.timeout > 0 else None
    study.optimize(objective, n_trials=args.n_trials, timeout=timeout)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Best trial: #{study.best_trial.number}")
    print(f"  Best loss:  {study.best_trial.value:.4f}")
    print("  Best params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")
    print(f"{'='*60}")

    # Save results
    results_path = Path(f"saves/{tag}/bayesian/search_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_data = {
        "best_trial": study.best_trial.number,
        "best_loss": study.best_trial.value,
        "best_params": study.best_trial.params,
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
            }
            for t in study.trials
        ],
    }
    results_path.write_text(json.dumps(results_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results saved to {results_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
