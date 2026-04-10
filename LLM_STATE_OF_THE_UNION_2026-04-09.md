# LLM State of the Union — 2026-04-09

> **One sentence:** A solo developer can now take a frontier-grade reasoning student from "nothing" to "training" in 60 seconds, or to "graduated and merged" overnight, using only public HuggingFace datasets and an editable LlamaFactory install.

This doc is the pocket guide to how the Distillation Studio actually works in April 2026 — who makes the data, who makes the software, what the moving parts are, and how to push the three buttons that matter: **Speed-Run**, **Overnight Graduation**, and the **Slow Path**.

---

## Part 1 — The kitchen analogy (read this first)

Forget "teacher/student/distillation" for a minute. Think of this whole ecosystem as a kitchen:

| Role | Who | What they do |
|:--|:--|:--|
| **Michelin chef** | Claude 4.6 Opus, Claude 4.5 Opus, DeepSeek-R1, Qwen3.5 | Cooks the hard, creative, expensive dishes. Can't come to your house. |
| **Food critics & sous-chefs** | `Roman1111111`, `nohurry`, `TeichAI`, `Jackrong` | Dined at the Michelin restaurant, wrote down exactly what the chef did (both the final plate and the *thinking while cooking*), published the recipes as free HF datasets. |
| **The cooking school** | `hiyouga/LlamaFactory` | The kitchen, the knives, the stoves, the YAML recipe binder. Turns "I have a dataset and a base model" into "I have a finetuned model". |
| **Trainee cooks** | Qwen2.5-0.5B / 1.5B / 3B / 7B, Gemma-4, Llama-3.2 | Your student. Small, fast, cheap to run locally. Learns by repeating the frontier recipes until it can cook them too. |
| **The repackagers** | `huihui-ai`, `mlx-community`, `TheBloke`-style quantizers | Take the finished dish, quantize to GGUF / MLX / AWQ / abliterated, make it run on a laptop. |
| **The head chef** | **You** | Pick the recipe book, pick the trainee, pick the oven, press Start. Come back in the morning to a graduated chef. |

**The whole magic of 2026 is that the sous-chefs already did the expensive part** (paying Anthropic ~$50,000 in API credits to generate `<think>...</think>` traces), so when you walk into the cooking school you don't need to hire a Michelin chef — you just need to read the recipe book and put it on the curriculum.

> **The wet dream of any LLM coder:** fire up a real reasoning-distilled training run in **under 60 seconds**, no teacher inference, no hallucination gates, no purification queue. As of today, LlamaFactory's Distillation Studio supports it natively as **Speed-Run mode**, and the new `scripts/speed_graduation.py` orchestrator lets you launch the whole overnight pipeline (download → import → configs → train → merge → report) with a single command.

---

## Part 2 — The three paths (pick your ambition)

There are three ways to train a student with this repo. Same code, same UI, same artifacts at the end — only the first couple of stages differ.

```
 ┌─────────────────────┐
 │   FRONTIER CHEF     │   Claude 4.6 Opus / DeepSeek-R1 / Qwen3.5
 │   (closed weights)  │
 └──────────┬──────────┘
            │ API $$$
            ▼
 ┌─────────────────────┐           ┌─────────────────┐
 │    SOUS-CHEFS       │◄──────────│ YOUR LOCAL      │
 │   (HF dataset       │           │ TEACHERS        │
 │   uploaders)        │           │ (slow path)     │
 └──────────┬──────────┘           └────────┬────────┘
            │                                │
       ┌────┴────┐                           │
       ▼         ▼                           ▼
  ┌────────┐ ┌──────────┐              ┌──────────┐
  │SPEED-  │ │OVERNIGHT │              │  SLOW    │
  │RUN     │ │SPEED     │              │  PATH    │
  │(60 s)  │ │GRADUATION│              │ (hours)  │
  │manual  │ │(one cmd) │              │ (full    │
  └────┬───┘ └─────┬────┘              │  control)│
       │           │                   └────┬─────┘
       └───────────┴───────────┬────────────┘
                               │
                               ▼
                   ┌─────────────────────┐
                   │  COOKING SCHOOL     │
                   │  (LlamaFactory)     │
                   │  SFT → DPO → merge  │
                   │  → GGUF → eval      │
                   └─────────────────────┘
                               │
                               ▼
                        ┌─────────────┐
                        │ GRADUATED   │
                        │ STUDENT     │
                        └─────────────┘
```

### Path A — **Speed-Run mode** (manual, in the browser, ≈60 s)

Open `python scripts/distill_server.py`, go to the Studio, check the **Speed-Run** box, pick a dataset, click **Start**. Under the hood, stages 1–3 (generate/halluc/purify) are replaced by a single call to `scripts/import_reasoning_dataset.py` which downloads the HF dataset and writes `consensus_sft.jsonl` directly. SFT starts within seconds. Best for: interactive iteration, tweaking a single dataset.

### Path B — **Overnight Speed Graduation** (one command, walk away) **← NEW**

```
python scripts/speed_graduation.py --tag overnight_grad
```

That's it. This is the "press one button before bed, wake up to a graduated model" path. It chains **download student → import dataset → register dataset → generate configs → CPU-safe patching → SFT train → LoRA merge → markdown report** with full resumability. If you Ctrl-C it, re-running with the same `--tag` skips completed stages. Best for: unattended overnight runs, CI, reproducible benchmarks.

### Path C — **Slow path** (full teacher pipeline, 1–10 hours)

The classic multi_teacher_generate → hallucination_gates → purify_teacher_outputs pipeline. Use this when you need to distill something that no HF sous-chef has published yet: proprietary domains (legal, medical, game lore), brand voice, internal documents, niche languages. Everything after stage 3 is identical to the Speed-Run path.

---

## Part 3 — The recipe book (HF datasets you can boot from)

These are the **pre-cooked Michelin traces** that make the Speed-Run path possible. Each has a converter inside `scripts/import_reasoning_dataset.py` so the Studio can download and normalize them automatically.

| Dataset | Size | Pedigree | Best for |
|:--|:--:|:--|:--|
| `Roman1111111/claude-opus-4.6-10000x` | ~9.6 k | Claude 4.6 Opus, messages-style | **Default for serious runs.** Biggest, most diverse, shareGPT format. |
| `nohurry/Opus-4.6-Reasoning-3000x-filtered` | ~2.3 k | Claude 4.6 Opus, pre-filtered | **Safest.** Separate problem/thinking/solution fields, already quality-gated. |
| `TeichAI/claude-4.5-opus-high-reasoning-250x` | ~250 | Claude 4.5 Opus, hand-picked | **Fastest sanity check.** 2-minute smoke test of a new student arch. |
| `Jackrong/Qwen3.5-reasoning-700x` | ~700 | Qwen-rephrased | Style ablation. Non-Claude baseline. |

All four are JSONL, all four are Apache/MIT-ish licensed by the uploader (with the usual caveat that the *underlying traces* were generated against a frontier lab's ToS — talk to a lawyer before commercial use).

### How to pick

- **Smoke-testing plumbing?** → TeichAI 250x (`--max-rows 250`)
- **Overnight run on a laptop?** → nohurry 3000x filtered (`--max-rows 2000`)
- **Best possible result, dedicated workstation?** → Roman 10000x (`--max-rows 8000+`)
- **Non-Claude style exploration?** → Jackrong 700x

---

## Part 4 — The cast of characters (who publishes what)

### The data harvesters (sous-chefs)

| Who | What they publish |
|:--|:--|
| **`Roman1111111`** | The 10k Opus 4.6 messages dataset. Largest public Opus trace collection as of 2026-04. |
| **`nohurry`** | Filtered, pre-quality-gated Opus 4.6 traces with separate thinking/solution fields. |
| **`TeichAI`** | Small, lovingly hand-picked Opus 4.5 and Gemma-4 reasoning sets. The "boutique" harvester. |
| **`Jackrong`** | Qwen-rephrased reasoning datasets + also publishes the distilled *models* (see below). |

### The trainers (other head chefs, for inspiration)

| Who | What they publish | Why it matters |
|:--|:--|:--|
| **`Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2`** | Already-trained 27B distilled student | The canonical benchmark: what's achievable with a serious box. Our `gen_distill_configs.py` defaults mirror their Unsloth + TRL recipe. |
| **`TeichAI/Gemma-4-*-distilled`** | Gemma-4 base + Opus traces | Non-Qwen reference point; proves the recipe transfers. |
| **`arsovskidev/Gemma-4-E4B-*`** | Gemma-4 E4B finetunes | Shows the tiny-base end of the spectrum. |

### The repackagers

| Who | What they do |
|:--|:--|
| **`huihui-ai`** | Abliterated / uncensored variants of popular distilled models. |
| **`mlx-community`** | 4-bit MLX quantizations for Apple Silicon. |
| **The GGUF community** (TheBloke-style) | Q4_K_M, Q5_K_M, Q8_0 quants for llama.cpp. Your `scripts/slim_down.py` does this in-house. |

### The cooking school

| Who | What they build |
|:--|:--|
| **`hiyouga/LlamaFactory`** | The framework this whole repo is built on. YAML-driven SFT/DPO/PPO/KTO/ORPO + 100+ model families + the `qwen3_5` template with `<think>` awareness we rely on. **If you only install one LLM training tool in 2026, install this one.** |
| **`Unsloth`** | 2× faster LoRA training via Triton kernels. The recipe we copy (Jackrong's) originated here. |
| **`TRL`** (HuggingFace) | The underlying `SFTTrainer`/`DPOTrainer` LlamaFactory wraps. |
| **`bitsandbytes`** | 8-bit AdamW + 4-bit base model quant. Enables 27B training on 24 GB. |

---

## Part 5 — The theory in five paragraphs

### 5.1 What "distillation" actually means in 2026

Modern LLM distillation is **teacher-student supervised finetuning on the teacher's full output trace**, chain-of-thought and all. The student sees `(prompt, <think>full_reasoning</think>final_answer)` pairs and learns to mimic not just the answer but the *shape* of the thinking. This is fundamentally different from pre-2024 distillation (which used teacher logits or hidden states) — in 2026 we distill on raw text, which means any base model can be a student and any text-completion API can be a teacher.

### 5.2 Why frontier traces beat your local teachers

Three moats:

1. **Compute moat.** A single Opus trace takes 30–120 s to generate on Anthropic's infra. Your 14B local teacher on a 24 GB GPU takes longer *and* produces lower-quality reasoning.
2. **RLHF moat.** Opus has months of RLHF behind it. Its mistakes are subtler; its correct answers are more likely to actually be correct.
3. **Distribution moat.** Frontier labs curate their training sets from real hard questions users send them. Your synthetic prompt set probably looks nothing like real user pain.

Sum: a 1.5B student trained on 5k Opus traces can outperform a 14B model trained on 100k self-generated traces on GSM8K / MATH / MMLU-Pro / BBH.

### 5.3 Why a tiny student can imitate a frontier chef

Naively this shouldn't work — Qwen2.5-1.5B has 1% the parameters of Claude. But:

- **Reasoning is mostly tokens, not parameters.** The `<think>` block is a serialized procedure. A small model has plenty of capacity to *imitate* a serialized procedure even if it couldn't have *invented* it.
- **The lift is inference-time, not weight-time.** When the student emits the right chain at inference, the final answer rides on the chain's shoulders. Same compute-for-quality tradeoff that made Chain-of-Thought work in the first place, just now it's learned instead of prompted.

### 5.4 What you give up

- **Domain coverage.** Upstream datasets are math/code/logic/science. Need a legal or proprietary-lore reasoner? Slow path with your own teachers.
- **Style alignment.** The student inherits Claude's voice (formal, hedged, cautiously helpful). For brand voice, add a second SFT pass.
- **License hygiene.** The upstream datasets are Apache/MIT by their uploaders, but the underlying outputs have a complicated provenance. Talk to a lawyer before commercial use.

### 5.5 When to use which path

| Goal | Path |
|:--|:--|
| Train a math/code reasoner in under 1 hour on a GPU | Speed-Run + Roman 10000x |
| Sanity-check a new student arch in 2 minutes | Speed-Run + TeichAI 250x |
| Overnight hands-off run on a CPU laptop | **`speed_graduation.py`** + nohurry 3000x |
| Build a domain reasoner (medical / legal / lore) | Slow path + your own teachers |
| Distill style from a model you already have | Slow path (load that model as teacher) |
| Reproduce Jackrong's 27B distilled release | Speed-Run with the full recipe |

---

## Part 6 — How to actually run it

### 6.1 Overnight Speed Graduation (recommended starting point)

The simplest possible command on a CPU-only laptop:

```bash
python scripts/speed_graduation.py --tag overnight_grad
```

**Defaults** (tuned for an unattended overnight run on the machine this doc was written on):

- Student: `Qwen/Qwen2.5-1.5B-Instruct`
- Dataset: `Roman1111111/claude-opus-4.6-10000x`
- `--max-rows 2000` (≈2000 GOLD samples, fits in one night of CPU wall-clock)
- `--epochs 1.0`
- `--cpu-safe` auto-detected (disables bf16, lowers cutoff to 4096, switches optim to `adamw_torch`)

**Alternate profiles:**

```bash
# 2-minute sanity check (0.5B student, 10 samples)
python scripts/speed_graduation.py --smoke-test \
    --tag smoke_grad \
    --student Qwen/Qwen2.5-0.5B-Instruct \
    --dataset TeichAI/claude-4.5-opus-high-reasoning-250x \
    --max-rows 10

# Ambitious GPU run (3B student, 5000 samples, 2 epochs)
python scripts/speed_graduation.py \
    --tag big_grad \
    --student Qwen/Qwen2.5-3B-Instruct \
    --dataset Roman1111111/claude-opus-4.6-10000x \
    --max-rows 5000 \
    --epochs 2 \
    --no-cpu-safe

# Dry-run (print the plan, execute nothing)
python scripts/speed_graduation.py --dry-run
```

**What gets written** on completion:

```
saves/<tag>/graduation_report.md          <-- human-readable summary
saves/<tag>/lora/sft/                     <-- LoRA adapter checkpoints
saves/<tag>/merged/                       <-- fully merged HF model, ready to load
saves/<tag>/speed_graduation_logs/*.log   <-- per-stage stdout
examples/distillation/auto/<tag>_sft.yaml <-- auto-generated training config
examples/distillation/auto/<tag>_merge.yaml
data/upstream_<tag>/consensus_sft.jsonl   <-- the GOLD samples
data/dataset_info.json                    <-- auto-registered so llamafactory-cli sees it
```

**Resumability.** Every stage writes a sentinel (a file or directory that didn't exist before). Re-running with the same `--tag` skips completed stages automatically. Ctrl-C during SFT? Re-run and LlamaFactory picks up from the last checkpoint. Ctrl-C after SFT? Re-run and it jumps straight to the merge stage.

### 6.2 Speed-Run mode in the browser Studio

```bash
python scripts/distill_server.py          # starts the local server
# open http://localhost:<port>/distill.html in a browser
```

Check the purple **Speed-Run** box, pick a dataset from the dropdown, set `max_rows`, click **Start**. Stages 1–3 in the progress rail will show "skip (upstream)" and SFT starts within seconds. Everything else in the Studio (curriculum, multi-student batches, postmortem, metrics rail) works unchanged.

### 6.3 Under the hood: what Speed-Run replaces

The slow path:

```
[generate]  multi_teacher_generate.py   --> 1-10h, 50-200% GPU util
[halluc]    hallucination_gates.py      --> 30-90m, 5-gate filter
[purify]    purify_teacher_outputs.py   --> 15-30m, majority vote -> GOLD/SILVER/DROP
```

Speed-Run:

```
[generate]  import_reasoning_dataset.py --> 5-60s, 1 HTTP download
[halluc]    <skipped>                   --> upstream is pre-filtered
[purify]    <skipped>                   --> everything is GOLD
```

Both produce the same shape on disk:

```
<output_dir>/consensus_sft.jsonl       <-- the GOLD samples
<output_dir>/purification_report.json  <-- gold/silver/drop counts
<output_dir>/teacher_responses.jsonl   <-- sentinel for pipeline_start()
<output_dir>/upstream_meta.json        <-- provenance (what was imported)
```

So stages 4–11 (configs, SFT, DPO, merge, GGUF, eval, dashboard) don't need to know or care which path was taken.

---

## Part 7 — The recipe knobs that matter (from Jackrong's playbook)

These defaults live in `scripts/gen_distill_configs.py`. They're tuned to match the recipe Jackrong published for his 27B distilled-v2 release (Unsloth + TRL SFTTrainer), then `speed_graduation.py` post-patches for CPU budgets.

| Knob | Default | Why |
|:--|:--|:--|
| `template` | `qwen3_5` | `ReasoningTemplate` class, knows about `<think>...</think>` spans. Critical. |
| `enable_thinking` | `True` | Don't strip the `<think>` blocks during tokenization. |
| `mask_history` | `True` | LlamaFactory's `train_on_responses_only`. Loss is computed **only on assistant turns**, not on the prompt. The single most important knob for style transfer. |
| `cutoff_len` | 8192 (GPU) / 4096 (CPU) | Reasoning traces commonly run 2k–8k tokens. A 1024 cutoff would **truncate every `<think>` block** and you'd learn garbage. |
| `lora_rank` | 32 | Reasoning style transfer needs more adapter capacity than vanilla SFT. Jackrong uses 64; 32 is our 16-GB-friendly compromise. |
| `lora_alpha` | 32 | Alpha = rank → scaling 1.0. |
| `learning_rate` | 5e-5 | Jackrong's 2e-4 is for Unsloth's faster convergence; 5e-5 is safer on plain TRL. |
| `num_train_epochs` | 2.0 (GPU) / 1.0 (CPU) | Reasoning transfer saturates quickly. Any more and you're memorizing. |
| `gradient_accumulation_steps` | 8 | Effective batch size = 8. On CPU this is the only lever to get stable gradients. |
| `optim` | `adamw_8bit` (GPU) / `adamw_torch` (CPU) | bitsandbytes 8-bit AdamW saves ~4× optimizer state memory, but only with a CUDA wheel. |
| `weight_decay` | 0.001 | Light regularization; Jackrong's value. |
| `warmup_ratio` | 0.05 | Slightly more warmup than the TRL default because LoRA is noisy early. |

---

## Part 8 — Hardware profiles

### 8.1 CPU-only laptop (the machine this was written on)

- **Student**: Qwen2.5-0.5B or 1.5B Instruct
- **Dataset**: nohurry 3000x filtered OR Roman 10000x with `--max-rows 2000`
- **Command**: `python scripts/speed_graduation.py --tag overnight_grad`
- **Expected wall-clock**: ~6–12 hours for 1.5B / 2000 samples / 1 epoch on a modern laptop CPU
- **Tradeoffs**: No bf16, cutoff capped at 4096, `adamw_torch` instead of `adamw_8bit`

### 8.2 Single-GPU workstation (16–24 GB VRAM)

- **Student**: Qwen2.5-3B or 7B Instruct
- **Dataset**: Roman 10000x, all of it (`--max-rows 10000`)
- **Command**: `python scripts/speed_graduation.py --tag gpu_grad --student Qwen/Qwen2.5-7B-Instruct --max-rows 10000 --no-cpu-safe`
- **Expected wall-clock**: 1–3 hours
- **Notes**: Enable `adamw_8bit` via bitsandbytes for memory headroom.

### 8.3 Multi-GPU / 48+ GB

- **Student**: Qwen2.5-14B or 32B / Gemma-4-27B
- **Dataset**: Roman 10000x (full) or two datasets concatenated
- **Path**: Full slow path recommended for maximum control, or Speed-Run with higher `lora_rank=64`.
- **Notes**: Match Jackrong's published recipe verbatim for reproducibility.

---

## Part 9 — The file map

New / changed in this release:

```
scripts/speed_graduation.py           NEW   the overnight one-click orchestrator
scripts/import_reasoning_dataset.py   NEW   HF dataset -> consensus_sft.jsonl converter
scripts/gen_distill_configs.py        UPD   Jackrong-recipe defaults (template, cutoff, LoRA)
scripts/distill_server.py             UPD   Speed-Run branch in pipeline_start()
scripts/distill.html                  UPD   purple Speed-Run card in the Studio UI
pyproject.toml                        UPD   requires-python >=3.10 (was >=3.14)
LLM_STATE_OF_THE_UNION_2026-04-09.md  NEW   this document
```

---

## Part 10 — FAQ

**Q: Can I add my own HF dataset to the Studio's Speed-Run dropdown?**
Yes. Either drop a converter into `scripts/import_reasoning_dataset.py::KNOWN`, or rely on the generic `_auto_convert` fallback (it handles `{messages}`, `{input,output}`, `{prompt,response}`, `{problem,thinking,solution}` shapes out of the box). Then add an `<option>` to the dropdown in `scripts/distill.html` and you're done.

**Q: Does Speed-Run skip DPO?**
Yes — upstream datasets don't contain the `chosen/rejected` pairs DPO needs. You get SFT-only from Speed-Run. If you want DPO on top, run Speed-Run first to get a strong SFT checkpoint, then run the slow path's purify stage (which produces `conflict_dpo.jsonl`) against a small set of your own teachers.

**Q: Is the merged model GGUF-ready?**
Yes. After `speed_graduation.py` finishes, run `python scripts/slim_down.py --model-dir saves/<tag>/merged --out-dir saves/<tag>/gguf --tag <tag> --quant q4_k_m` to produce a `.gguf` you can load in llama.cpp / LM Studio / llama-server.

**Q: What if the student download fails?**
`speed_graduation.py` uses `huggingface_hub.snapshot_download()` with an explicit `allow_patterns` whitelist. If your HF cache is at a non-default path, set `HF_HOME` before running (the machine this was written on uses `HF_HOME=C:\AI\Models\.hf_cache`).

**Q: Can I resume after a crash?**
Yes. Re-run with the same `--tag` and every stage that already has its sentinel on disk will be skipped. The most common recovery is: crash during SFT → re-run → SFT resumes from the last saved checkpoint (LlamaFactory handles this natively).

**Q: What if I want to train multiple students on the same GOLD set?**
Two options: (1) the Studio's multi-student batch mode (runs stages 1–3 once, loops stages 4–11 per student), or (2) run `speed_graduation.py` N times with different `--tag` and `--student` — the dataset import is a no-op after the first run because `upstream_<tag>` is cached.

**Q: Is this legal?**
The datasets are Apache/MIT by their uploaders. The underlying Opus traces were generated against Anthropic's API — the provenance is complicated. For personal research, fine. For commercial use, talk to a lawyer. Not legal advice.

**Q: What does LlamaFactory give me that a raw TRL script doesn't?**
YAML-driven configs (no glue code), 100+ pretrained model families with correct templates, built-in `train_on_responses_only`/`mask_history`, unified SFT/DPO/PPO/KTO/ORPO interface, LoRA/QLoRA/full-finetune/PiSSA/LoftQ/DoRA in one binary, and the `qwen3_5` template with `<think>` awareness. It's the difference between writing ten 500-line training scripts and writing ten 30-line YAMLs.

---

## Part 11 — Where to go next

1. **Run the overnight graduation.** `python scripts/speed_graduation.py --tag overnight_grad`. Walk away. Come back in the morning.
2. **Read the graduation report.** `saves/overnight_grad/graduation_report.md` — stage timings, artifact paths, next-step commands.
3. **Quantize to GGUF.** `python scripts/slim_down.py --model-dir saves/overnight_grad/merged ...`
4. **Run the eval panel.** `python scripts/eval_student_panel.py --saves-tag overnight_grad --probes data/eval_probes.jsonl`
5. **Compare to Jackrong's published 27B benchmark.** Use it as your ceiling.

---

*— 2026-04-09, written at the edge of what open-weights distillation can do.*
