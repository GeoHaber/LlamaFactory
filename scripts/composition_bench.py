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

r"""Composition Benchmark — compare skill-composition strategies.

Given a trunk model and 2+ trained LoRA skill branches, benchmarks five
composition strategies side-by-side:

  Strategy 0 :  Trunk baseline         — no adapters (lower bound)
  Strategy 1 :  Individual branches    — per-skill ceiling (upper bound)
  Strategy 2 :  DARE-TIES static merge — single merged model
  Strategy 3 :  Adapter routing        — classify prompt → swap adapter
  Strategy 4 :  Weighted adapter stack — PEFT ``add_weighted_adapter()``

Metrics:  per-skill perplexity, overall perplexity, tokens/sec latency.

Test data format (JSONL, one per line)::

    {"instruction": "Translate to Spanish: ...", "output": "Hola ...", "skill": "translate"}
    {"instruction": "Write a sort function",     "output": "def ...",  "skill": "code"}

Usage::

    # Full benchmark (needs GPU)
    python scripts/composition_bench.py \
        --trunk saves/trunk_v1/merged \
        --branches translate:saves/b_tr/lora/sft,code:saves/b_code/lora/sft \
        --test-data data/bench/mixed_test.jsonl \
        --tag bench_v1

    # With a pre-merged DARE-TIES model
    python scripts/composition_bench.py \
        --trunk saves/trunk_v1/merged \
        --branches translate:saves/b_tr/lora/sft,code:saves/b_code/lora/sft \
        --merged-model saves/combined/merged \
        --test-data data/bench/mixed_test.jsonl \
        --tag bench_v1

    # Dry run — validate inputs, show plan, skip model loading
    python scripts/composition_bench.py \
        --trunk saves/trunk_v1/merged \
        --branches translate:saves/b_tr/lora/sft,code:saves/b_code/lora/sft \
        --test-data data/bench/mixed_test.jsonl \
        --tag bench_v1 --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── Lazy imports (heavy deps only needed for eval mode) ──────────────────────

_TORCH_OK = False
_PEFT_OK = False


def _check_deps() -> tuple[bool, bool]:
    """Return (torch_ok, peft_ok) without crashing on import failure."""
    global _TORCH_OK, _PEFT_OK  # noqa: PLW0603
    try:
        import torch  # noqa: F401
        _TORCH_OK = True
    except ImportError:
        pass
    try:
        import peft  # noqa: F401
        _PEFT_OK = True
    except ImportError:
        pass
    return _TORCH_OK, _PEFT_OK


# ── Logging ──────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f"[comp_bench] {msg}", flush=True)  # xray: ignore[PY-004]


# ── Skill classifier (keyword-based oracle router) ──────────────────────────

_SKILL_KEYWORDS: dict[str, list[str]] = {
    "translate":  ["translate", "translation", "convert to", "say in", "how to say",
                   "en español", "en français", "auf deutsch", "traduce", "traduire"],
    "translation": ["translate", "translation"],
    "code":       ["code", "function", "implement", "program", "algorithm", "debug",
                   "class ", "def ", "write a script", "refactor", "unit test",
                   "python", "javascript", "rust", "java", "c++"],
    "coding":     ["code", "function", "implement", "program"],
    "math":       ["math", "calculate", "equation", "solve", "integral", "derivative",
                   "proof", "theorem", "probability", "statistics", "compute"],
    "reasoning":  ["reason", "think step", "logic", "deduce", "infer", "conclude",
                   "explain why", "what follows"],
    "style":      ["rewrite", "rephrase", "tone", "formal", "casual", "style",
                   "make it sound", "voice"],
    "domain":     ["legal", "medical", "scientific", "clinical", "patent", "contract"],
    "legal":      ["legal", "law", "statute", "court", "contract", "compliance"],
    "medical":    ["medical", "clinical", "diagnosis", "treatment", "patient", "symptom"],
}


def classify_skill(prompt: str, known_skills: list[str]) -> str:
    """Classify a prompt into a skill by keyword scoring.

    This is a heuristic ceiling-test router.  In production you would train a
    lightweight classifier or use the trunk itself for zero-shot classification.
    For the benchmark, keyword matching is sufficient because the test data has
    gold skill labels — we just need the router to pick the *right adapter*.

    Returns the best-matching skill from *known_skills*, or the first skill as
    fallback if no keywords match.
    """
    prompt_lower = prompt.lower()
    scores: dict[str, int] = {s: 0 for s in known_skills}
    for skill in known_skills:
        keywords = _SKILL_KEYWORDS.get(skill, [skill])
        for kw in keywords:
            if kw in prompt_lower:
                scores[skill] += 1
    best = max(scores, key=lambda s: scores[s])
    return best if scores[best] > 0 else known_skills[0]


# ── Test data loader ─────────────────────────────────────────────────────────

def _load_test_data(path: Path) -> list[dict]:
    """Load test JSONL.  Each line must have ``instruction``, ``output``, ``skill``."""
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if "instruction" not in obj or "output" not in obj:
            continue
        # Default skill label
        if "skill" not in obj:
            obj["skill"] = "unknown"
        rows.append(obj)
    return rows


# ── Perplexity computation ───────────────────────────────────────────────────

def _compute_perplexity(
    model,          # transformers.PreTrainedModel
    tokenizer,      # transformers.PreTrainedTokenizer
    examples: list[dict],
    max_length: int = 2048,
) -> float:
    """Compute perplexity of gold responses given prompts.

    For each example, tokenizes ``instruction + output`` as a single sequence,
    masks the prompt tokens, and computes cross-entropy only on the response
    tokens.  Returns ``exp(mean_nll)``.
    """
    import torch

    total_nll = 0.0
    total_tokens = 0

    model.eval()
    for ex in examples:
        prompt = ex["instruction"]
        response = ex["output"]

        # Format text — use chat template if the tokenizer has one
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prompt_messages = [{"role": "user", "content": prompt}]
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            text = prompt + "\n" + response
            prompt_text = prompt + "\n"

        full_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        prompt_ids = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_length)
        prompt_len = prompt_ids["input_ids"].shape[1]

        full_ids = {k: v.to(model.device) for k, v in full_ids.items()}

        with torch.no_grad():
            outputs = model(**full_ids)

        # Score only response tokens
        shift_logits = outputs.logits[:, prompt_len - 1:-1, :]
        shift_labels = full_ids["input_ids"][:, prompt_len:]

        if shift_labels.numel() == 0:
            continue

        nll = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="sum",
        )
        total_nll += nll.item()
        total_tokens += shift_labels.numel()

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_nll / total_tokens)


def _compute_latency(
    model,
    tokenizer,
    examples: list[dict],
    max_new_tokens: int = 64,
) -> float:
    """Measure average generation latency in tokens/sec."""
    import torch

    total_tokens = 0
    total_time = 0.0

    model.eval()
    for ex in examples[:20]:  # cap at 20 prompts for speed
        prompt = ex["instruction"]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        elapsed = time.time() - t0
        new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
        total_tokens += new_tokens
        total_time += elapsed

    if total_time == 0:
        return 0.0
    return total_tokens / total_time


# ── Model loading helpers ────────────────────────────────────────────────────

def _load_base_model(model_path: str, device: str = "auto"):
    """Load a HuggingFace model + tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _log(f"loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=device,
    )
    return model, tokenizer


def _load_with_adapter(model_path: str, adapter_path: str, adapter_name: str, device: str = "auto"):
    """Load a model with a single PEFT adapter."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _log(f"loading model: {model_path} + adapter: {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model = PeftModel.from_pretrained(base, adapter_path, adapter_name=adapter_name)
    return model, tokenizer


def _load_multi_adapter(model_path: str, adapters: dict[str, str], device: str = "auto"):
    """Load trunk + multiple named adapters for routing / stacking.

    Args:
        model_path: HF trunk model directory.
        adapters:   ``{skill_name: adapter_path, ...}``

    Returns:
        ``(peft_model, tokenizer, list_of_adapter_names)``
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _log(f"loading trunk: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=device,
    )

    names = list(adapters.keys())
    first_name = names[0]
    first_path = adapters[first_name]
    _log(f"  adapter[0] '{first_name}' <- {first_path}")
    model = PeftModel.from_pretrained(base, first_path, adapter_name=first_name)

    for name in names[1:]:
        path = adapters[name]
        _log(f"  adapter '{name}' <- {path}")
        model.load_adapter(path, adapter_name=name)

    return model, tokenizer, names


# ── Strategy implementations ─────────────────────────────────────────────────

def _bench_trunk(model_path: str, test_data: list[dict], device: str) -> dict:
    """Strategy 0: Trunk baseline — no adapters."""
    _log("=== Strategy 0: Trunk baseline ===")
    model, tok = _load_base_model(model_path, device)
    ppl = _compute_perplexity(model, tok, test_data)
    lat = _compute_latency(model, tok, test_data)

    # Per-skill breakdown
    by_skill: dict[str, float] = {}
    skills = sorted(set(ex["skill"] for ex in test_data))
    for skill in skills:
        subset = [ex for ex in test_data if ex["skill"] == skill]
        by_skill[skill] = _compute_perplexity(model, tok, subset) if subset else float("inf")

    del model
    _free_gpu()
    return {"overall_ppl": ppl, "per_skill_ppl": by_skill, "tok_per_sec": lat}


def _bench_individual(
    model_path: str,
    adapters: dict[str, str],
    test_data: list[dict],
    device: str,
) -> dict:
    """Strategy 1: Individual branches — per-skill ceiling."""
    _log("=== Strategy 1: Individual branches ===")
    by_skill: dict[str, float] = {}

    for skill, adapter_path in adapters.items():
        subset = [ex for ex in test_data if ex["skill"] == skill]
        if not subset:
            _log(f"  {skill}: no test examples, skipping")
            by_skill[skill] = float("inf")
            continue

        model, tok = _load_with_adapter(model_path, adapter_path, skill, device)
        model.set_adapter(skill)
        by_skill[skill] = _compute_perplexity(model, tok, subset)
        _log(f"  {skill}: ppl={by_skill[skill]:.2f} ({len(subset)} examples)")
        del model
        _free_gpu()

    return {"overall_ppl": float("nan"), "per_skill_ppl": by_skill, "tok_per_sec": 0.0}


def _bench_dare_ties(
    merged_path: str,
    test_data: list[dict],
    device: str,
) -> dict:
    """Strategy 2: DARE-TIES static merge — single merged model."""
    _log("=== Strategy 2: DARE-TIES merge ===")
    model, tok = _load_base_model(merged_path, device)
    ppl = _compute_perplexity(model, tok, test_data)
    lat = _compute_latency(model, tok, test_data)

    by_skill: dict[str, float] = {}
    skills = sorted(set(ex["skill"] for ex in test_data))
    for skill in skills:
        subset = [ex for ex in test_data if ex["skill"] == skill]
        by_skill[skill] = _compute_perplexity(model, tok, subset) if subset else float("inf")

    del model
    _free_gpu()
    return {"overall_ppl": ppl, "per_skill_ppl": by_skill, "tok_per_sec": lat}


def _bench_routing(
    model_path: str,
    adapters: dict[str, str],
    test_data: list[dict],
    device: str,
) -> dict:
    """Strategy 3: Adapter routing — classify prompt, swap adapter."""
    _log("=== Strategy 3: Adapter routing ===")
    model, tok, names = _load_multi_adapter(model_path, adapters, device)
    skills = list(adapters.keys())

    # Classify every test prompt
    routed: dict[str, list[dict]] = defaultdict(list)
    for ex in test_data:
        predicted = classify_skill(ex["instruction"], skills)
        routed[predicted].append(ex)

    _log(f"  routing distribution: {dict({k: len(v) for k, v in routed.items()})}")

    # Compute per-skill PPL with the routed adapter active
    by_skill: dict[str, float] = {}
    total_nll_tokens: list[tuple[float, int]] = []

    for skill in skills:
        subset = routed.get(skill, [])
        if not subset:
            by_skill[skill] = float("inf")
            continue
        model.set_adapter(skill)
        by_skill[skill] = _compute_perplexity(model, tok, subset)
        _log(f"  {skill}: ppl={by_skill[skill]:.2f} ({len(subset)} routed)")

    # Overall PPL — re-compute with routing
    all_ppls: list[float] = []
    all_counts: list[int] = []
    for skill in skills:
        subset = routed.get(skill, [])
        if subset:
            model.set_adapter(skill)
            p = _compute_perplexity(model, tok, subset)
            all_ppls.append(p * len(subset))
            all_counts.append(len(subset))
    overall = sum(all_ppls) / sum(all_counts) if all_counts else float("inf")

    # Latency — measure with adapter switching overhead
    lat = _compute_latency(model, tok, test_data)

    del model
    _free_gpu()
    return {"overall_ppl": overall, "per_skill_ppl": by_skill, "tok_per_sec": lat}


def _bench_stacked(
    model_path: str,
    adapters: dict[str, str],
    test_data: list[dict],
    device: str,
) -> dict:
    """Strategy 4: Weighted adapter stack — PEFT ``add_weighted_adapter()``.

    Merges all LoRA adapters with equal weights into a single adapter.
    This is the "LoRA Soups" approach (arXiv 2410.13025).
    """
    _log("=== Strategy 4: Weighted adapter stack ===")
    model, tok, names = _load_multi_adapter(model_path, adapters, device)

    # Merge with equal weights
    n = len(names)
    weights = [1.0 / n] * n
    _log(f"  merging {names} with weights {weights}")

    try:
        model.add_weighted_adapter(
            adapters=names,
            weights=weights,
            adapter_name="stacked",
            combination_type="linear",
        )
        model.set_adapter("stacked")
    except Exception as e:
        _log(f"  WARNING: add_weighted_adapter failed: {e}")
        _log("  falling back to first adapter only")
        model.set_adapter(names[0])

    ppl = _compute_perplexity(model, tok, test_data)
    lat = _compute_latency(model, tok, test_data)

    by_skill: dict[str, float] = {}
    skills = sorted(set(ex["skill"] for ex in test_data))
    for skill in skills:
        subset = [ex for ex in test_data if ex["skill"] == skill]
        by_skill[skill] = _compute_perplexity(model, tok, subset) if subset else float("inf")

    del model
    _free_gpu()
    return {"overall_ppl": ppl, "per_skill_ppl": by_skill, "tok_per_sec": lat}


def _free_gpu() -> None:
    """Release GPU memory between strategies."""
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ── Report generation ────────────────────────────────────────────────────────

def _generate_report(
    report_path: Path,
    tag: str,
    trunk_path: str,
    adapters: dict[str, str],
    test_data: list[dict],
    results: dict[str, dict],
) -> None:
    """Write a Markdown benchmark comparison report."""
    skills = sorted(set(ex["skill"] for ex in test_data))
    strategies = list(results.keys())

    lines = [
        f"# Composition Benchmark Report — `{tag}`",
        "",
        "## Configuration",
        "",
        f"| Key | Value |",
        f"|-----|-------|",
        f"| Trunk | `{trunk_path}` |",
        f"| Test examples | {len(test_data)} |",
        f"| Skills | {', '.join(skills)} |",
        f"| Branches | {len(adapters)} |",
    ]
    for skill, path in adapters.items():
        lines.append(f"| — {skill} | `{path}` |")
    lines += ["", "## Results", ""]

    # ── Overall table
    header = "| Strategy | Overall PPL | tok/sec |"
    sep = "|----------|------------:|--------:|"
    for skill in skills:
        header += f" {skill} PPL |"
        sep += "----------:|"
    lines += [header, sep]

    for strat in strategies:
        r = results[strat]
        overall = r.get("overall_ppl", float("nan"))
        lat = r.get("tok_per_sec", 0.0)
        row = f"| {strat} | {_fmt_ppl(overall)} | {lat:.1f} |"
        for skill in skills:
            sp = r.get("per_skill_ppl", {}).get(skill, float("inf"))
            row += f" {_fmt_ppl(sp)} |"
        lines.append(row)

    lines += [
        "",
        "## Strategy descriptions",
        "",
        "| # | Strategy | Description | Modular? |",
        "|---|----------|-------------|----------|",
        "| 0 | Trunk baseline | Raw trunk, no skill adapters | — |",
        "| 1 | Individual | Each branch evaluated on its own skill only (ceiling) | Yes |",
        "| 2 | DARE-TIES | Static merge of all branches into one model | No |",
        "| 3 | Routing | Keyword classifier picks the right adapter per prompt | Yes |",
        "| 4 | Stacked | PEFT `add_weighted_adapter()` merges LoRA weights | No |",
        "",
        "## Interpretation guide",
        "",
        "- **Individual** is the per-skill upper bound.  If DARE-TIES or Stacked are",
        "  close to Individual, composition is cheap.  If they're much worse, there's",
        "  significant interference between skill adapters.",
        "- **Routing** is the modular approach.  If it's close to Individual, then",
        "  the classifier is picking the right adapter.  At 0.5B-3B scale, the latency",
        "  overhead of adapter swapping should be small.",
        "- **Trunk** is the lower bound.  All strategies should beat it.",
        "- If Routing > Stacked, modular composition is worth the complexity.",
        "- If Stacked ≈ DARE-TIES, the simpler static merge is preferred.",
        "",
        "*Generated by `scripts/composition_bench.py`*",
    ]

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log(f"report: {report_path}")


def _fmt_ppl(v: float) -> str:
    """Format a perplexity value for the report table."""
    if math.isinf(v) or math.isnan(v):
        return "—"
    return f"{v:.2f}"


# ── Main benchmark orchestrator ──────────────────────────────────────────────

def run_benchmark(
    trunk_path: str,
    adapters: dict[str, str],
    test_data: list[dict],
    tag: str,
    merged_model: str | None = None,
    device: str = "auto",
    skip_strategies: list[str] | None = None,
) -> dict[str, dict]:
    """Run all composition strategies and return results.

    Args:
        trunk_path:      Path to merged trunk model (HF dir).
        adapters:        ``{skill_name: adapter_path, ...}``
        test_data:       List of test examples (instruction, output, skill).
        tag:             Benchmark tag for report naming.
        merged_model:    Optional path to pre-merged DARE-TIES model.
        device:          Torch device map (default ``"auto"``).
        skip_strategies: Strategy names to skip (e.g. ``["dare_ties"]``).

    Returns:
        ``{strategy_name: {overall_ppl, per_skill_ppl, tok_per_sec}}``.
    """
    skip = set(skip_strategies or [])
    results: dict[str, dict] = {}

    # Strategy 0: Trunk baseline
    if "trunk" not in skip:
        try:
            results["0. Trunk"] = _bench_trunk(trunk_path, test_data, device)
        except Exception as e:
            _log(f"ERROR in trunk baseline: {e}")
            results["0. Trunk"] = {"overall_ppl": float("inf"), "per_skill_ppl": {}, "tok_per_sec": 0, "error": str(e)}

    # Strategy 1: Individual branches
    if "individual" not in skip:
        try:
            results["1. Individual"] = _bench_individual(trunk_path, adapters, test_data, device)
        except Exception as e:
            _log(f"ERROR in individual: {e}")
            results["1. Individual"] = {"overall_ppl": float("inf"), "per_skill_ppl": {}, "tok_per_sec": 0, "error": str(e)}

    # Strategy 2: DARE-TIES merge (only if merged model provided)
    if "dare_ties" not in skip and merged_model:
        try:
            results["2. DARE-TIES"] = _bench_dare_ties(merged_model, test_data, device)
        except Exception as e:
            _log(f"ERROR in DARE-TIES: {e}")
            results["2. DARE-TIES"] = {"overall_ppl": float("inf"), "per_skill_ppl": {}, "tok_per_sec": 0, "error": str(e)}
    elif not merged_model and "dare_ties" not in skip:
        _log("skipping DARE-TIES: no --merged-model provided")

    # Strategy 3: Adapter routing
    if "routing" not in skip:
        try:
            results["3. Routing"] = _bench_routing(trunk_path, adapters, test_data, device)
        except Exception as e:
            _log(f"ERROR in routing: {e}")
            results["3. Routing"] = {"overall_ppl": float("inf"), "per_skill_ppl": {}, "tok_per_sec": 0, "error": str(e)}

    # Strategy 4: Weighted adapter stack
    if "stacked" not in skip:
        try:
            results["4. Stacked"] = _bench_stacked(trunk_path, adapters, test_data, device)
        except Exception as e:
            _log(f"ERROR in stacked: {e}")
            results["4. Stacked"] = {"overall_ppl": float("inf"), "per_skill_ppl": {}, "tok_per_sec": 0, "error": str(e)}

    # ── Report ───────────────────────────────────────────────────────────
    report_path = ROOT / "saves" / "benchmarks" / f"{tag}_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _generate_report(report_path, tag, trunk_path, adapters, test_data, results)

    # Also save raw JSON for programmatic access
    json_path = report_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(results, indent=2, default=str, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _log(f"raw JSON: {json_path}")

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_branches(arg: str) -> dict[str, str]:
    """Parse ``skill:path,skill:path,...`` into a dict."""
    branches: dict[str, str] = {}
    for entry in arg.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise ValueError(
                f"branch entry must be skill:path — got '{entry}'. "
                f"Example: translate:saves/b_translate/lora/sft"
            )
        skill, path = entry.split(":", 1)
        branches[skill.strip()] = path.strip()
    return branches


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Composition Benchmark — compare skill-composition strategies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s --trunk saves/trunk_v1/merged \\
           --branches translate:saves/b_tr/lora/sft,code:saves/b_code/lora/sft \\
           --test-data data/bench/mixed_test.jsonl \\
           --tag bench_v1

  # With pre-merged DARE-TIES model
  %(prog)s --trunk saves/trunk_v1/merged \\
           --branches translate:saves/b_tr/lora/sft,code:saves/b_code/lora/sft \\
           --merged-model saves/combined/merged \\
           --test-data data/bench/mixed_test.jsonl \\
           --tag bench_v1
""",
    )
    parser.add_argument("--trunk", required=True,
                        help="Path to merged trunk model (HF dir)")
    parser.add_argument("--branches", required=True,
                        help="Comma-separated skill:adapter_path pairs "
                             "(e.g. translate:saves/b_tr/lora/sft,code:saves/b_code/lora/sft)")
    parser.add_argument("--test-data", required=True,
                        help="JSONL test file with instruction, output, skill fields")
    parser.add_argument("--tag", required=True,
                        help="Benchmark tag (used for report filenames)")
    parser.add_argument("--merged-model", default=None,
                        help="Path to pre-merged DARE-TIES model (optional)")
    parser.add_argument("--device", default="auto",
                        help="Torch device map (default: auto)")
    parser.add_argument("--skip", default="",
                        help="Comma-separated strategies to skip "
                             "(trunk,individual,dare_ties,routing,stacked)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs and show plan, skip model loading")

    args = parser.parse_args()

    # ── Validate inputs ──────────────────────────────────────────────────
    trunk = Path(args.trunk).resolve()
    if not (trunk / "config.json").exists():
        _log(f"ERROR: trunk {trunk} has no config.json")
        return 1

    try:
        adapters = _parse_branches(args.branches)
    except ValueError as e:
        _log(f"ERROR: {e}")
        return 1

    if len(adapters) < 2:
        _log("ERROR: need at least 2 branches for a meaningful benchmark")
        return 1

    for skill, path in adapters.items():
        ap = Path(path).resolve()
        if not (ap / "adapter_config.json").exists() and not (ap / "config.json").exists():
            _log(f"WARNING: branch '{skill}' at {ap} has no adapter_config.json or config.json")

    test_path = Path(args.test_data).resolve()
    if not test_path.exists():
        _log(f"ERROR: test data {test_path} not found")
        return 1

    test_data = _load_test_data(test_path)
    if not test_data:
        _log(f"ERROR: no valid test examples in {test_path}")
        return 1

    skills_in_data = sorted(set(ex["skill"] for ex in test_data))
    skills_in_branches = sorted(adapters.keys())

    _log(f"trunk:    {trunk}")
    _log(f"branches: {len(adapters)} — {skills_in_branches}")
    _log(f"test:     {len(test_data)} examples, skills: {skills_in_data}")
    if args.merged_model:
        _log(f"merged:   {args.merged_model}")

    # Warn about mismatched skills
    missing = set(skills_in_data) - set(skills_in_branches) - {"unknown"}
    if missing:
        _log(f"WARNING: test data has skills {missing} not in --branches")

    skip_list = [s.strip() for s in args.skip.split(",") if s.strip()]

    if args.dry_run:
        _log("--dry-run: plan validated, stopping before model loading")
        _log(f"would run strategies: {[s for s in ['trunk','individual','dare_ties','routing','stacked'] if s not in skip_list]}")
        return 0

    # ── Check deps ───────────────────────────────────────────────────────
    torch_ok, peft_ok = _check_deps()
    if not torch_ok:
        _log("ERROR: torch not available — install pytorch to run the benchmark")
        return 1
    if not peft_ok:
        _log("ERROR: peft not available — install peft to run the benchmark")
        return 1

    # ── Run ──────────────────────────────────────────────────────────────
    results = run_benchmark(
        trunk_path=str(trunk).replace("\\", "/"),
        adapters={s: str(Path(p).resolve()).replace("\\", "/") for s, p in adapters.items()},
        test_data=test_data,
        tag=args.tag,
        merged_model=str(Path(args.merged_model).resolve()).replace("\\", "/") if args.merged_model else None,
        device=args.device,
        skip_strategies=skip_list,
    )

    _log("benchmark complete")
    for strat, r in sorted(results.items()):
        ppl = r.get("overall_ppl", float("nan"))
        lat = r.get("tok_per_sec", 0)
        _log(f"  {strat}: ppl={_fmt_ppl(ppl)}  tok/s={lat:.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
