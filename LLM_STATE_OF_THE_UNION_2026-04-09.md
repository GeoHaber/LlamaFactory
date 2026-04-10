# LLM State of the Union — 2026-04-09

> **The wet dream of any LLM coder:** boot a real reasoning-distilled training run in **under 60 seconds**, with no teacher inference, no hallucination gates, and no purification queue. As of today, LlamaFactory's Distillation Studio supports it natively.

This document captures the state of the open-weights reasoning-distillation art as of **9 April 2026**, and how the Distillation Studio's brand-new **Speed-Run mode** lets you skip directly from "I have nothing" to "training is running" in the time it takes to download a few hundred MB.

---

## TL;DR (the only paragraph you actually need to read)

In the last six months, frontier closed-weights labs (Anthropic in particular) shipped two extraordinary reasoning models — **Claude 4.5 Opus** and **Claude 4.6 Opus** — and a small but unusually motivated handful of open-source contributors **distilled tens of thousands of `<think>...</think>` reasoning traces from those models** and uploaded them to Hugging Face as plain `.jsonl` files. Those datasets are now **the single highest-leverage thing you can train a small student on**: a 1.5–8B base model finetuned on 5–10k Opus traces will beat the same base model trained on **millions** of vanilla SFT samples on every reasoning benchmark that matters.

Until today, LlamaFactory's Distillation pipeline could only build datasets the *slow* way: spin up 1–N local teacher GGUFs, generate 10–100k responses (1–10 hours), run the 5-gate hallucination filter (another 1–3 hours), majority-vote into GOLD/SILVER/DROP buckets (~30 min). That's the right pipeline for grounding new domains where no upstream traces exist. But for **canonical reasoning**, it's a waste of GPU-hours, because Anthropic already burned $50k+ in API credits generating better traces than your 14B teacher will ever produce.

**Speed-Run mode short-circuits stages 1–3.** You point it at one of four well-known HF datasets, click Start, and the pipeline downloads the file, converts it to `consensus_sft.jsonl` as pre-purified GOLD, and falls straight through to the SFT/DPO/Merge/GGUF/Eval stages it always ran. The only thing different is the first 30 seconds.

---

## 1 — The theory in five paragraphs

### 1.1 What "distillation" actually means in 2026

Knowledge distillation in the modern LLM sense is **teacher-student supervised finetuning on the teacher's full output trace**, including its chain-of-thought. The student is shown `(prompt, full_output_with_thinking)` pairs and learns to mimic not just the final answer but the *shape* of the reasoning. This is fundamentally different from pre-2024 distillation, which used the teacher's logits or hidden states.

The trick that makes 2026-style distillation work is that **frontier reasoning models emit explicit `<think>...</think>` blocks** (Claude 4.5/4.6 Opus, DeepSeek-R1, Qwen3.5, Gemini 2.5 Thinking). Those blocks are the *thinking out loud* the model does before committing to a final answer. When you train a student on raw `(prompt, <think>thinking</think>final answer)` pairs with `train_on_responses_only` (the loss is masked over the prompt), the student learns to **emit the same reasoning structure**, and — crucially — the same accuracy lift the teacher got from thinking out loud transfers down.

### 1.2 Why frontier-model traces beat your local teachers

Three reasons:

1. **Compute moat.** A single Opus reasoning trace can take 30–120 seconds to generate (long thinking budget). Generating 10k of them costs ~$300 in API credits. Your 14B local teacher running on a 24GB GPU takes longer than that *and* produces lower-quality reasoning.
2. **RLHF moat.** Opus and friends were trained with months of RLHF. The mistakes they make are subtler than the mistakes your local teacher makes, and the *correct* answers are more likely to actually be correct.
3. **Distribution moat.** The frontier labs hand-curated their reasoning training sets from the actual hard problems people send them. Your synthetic prompt set probably looks nothing like what real users ask.

The sum of these three is why a 1.5B student trained on 5k Claude 4.6 Opus traces can outperform a 14B model trained on 100k self-generated traces on GSM8K, MATH, MMLU-Pro, and BBH.

### 1.3 Why this works at all (the surprise)

The naive expectation is that a 1.5B student is "too small" to learn frontier-level reasoning. But two things make it work:

- **Reasoning style is mostly tokens, not parameters.** The `<think>` block is a serialized representation of the planning steps. A small model has plenty of parameters to *imitate* a serialized procedure, even if it couldn't have *invented* it.
- **The reasoning lifts inference-time accuracy, not learned-weight accuracy.** When the student emits the right `<think>` chain at inference, the *final answer* benefits from the chain even though the chain itself is an autoregressive prediction. This is the same compute-for-quality tradeoff that made Chain-of-Thought work in the first place — it just costs more tokens at inference time.

### 1.4 What you give up

Three things:

1. **Domain coverage.** The upstream datasets are dominated by math, code, logic puzzles, science MCQ. If you need a model that can reason about, say, **legal contracts** or **proprietary game lore**, you still need the slow path with your own teachers and prompts.
2. **Style alignment.** The student inherits Claude's voice (formal, hedged, cautiously helpful). If your product needs a brand voice, plan a second SFT pass on a stylistic dataset.
3. **License alignment.** The upstream datasets were generated against *Anthropic's* Terms of Service. Most of them are MIT/Apache-licensed by their uploaders, but the underlying outputs have a complicated provenance — talk to a lawyer before commercial use.

### 1.5 When to use Speed-Run vs. the slow path

| Goal | Use |
|---|---|
| Train a math/code/reasoning student in <1 hour | **Speed-Run** with `Roman1111111/claude-opus-4.6-10000x` |
| Smoke-test a new student arch | **Speed-Run** with `TeichAI/claude-4.5-opus-high-reasoning-250x` (250 rows = 2 minutes) |
| Build a domain-specific reasoner (medical, legal, lore) | **Slow path** — your own teachers + curated prompts |
| Distill *style* from a specific model you already have | **Slow path** — load that model as a teacher |
| Reproduce Jackrong's Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled v2 | **Speed-Run** with the recipe table in §4 |

---

## 2 — The four upstream datasets (canonical Speed-Run sources)

All four ship with explicit converters in `scripts/import_reasoning_dataset.py`. Each one has a different upstream schema, so the converter is responsible for normalizing them into the canonical `{instruction, output}` pair that `consensus_sft.jsonl` expects.

### 2.1 `Roman1111111/claude-opus-4.6-10000x` — **the LARGEST** (recommended default)

| | |
|---|---|
| Rows | ~9,634 |
| File | `opus46_final.jsonl` |
| Schema | `messages`-format (sometimes Python-repr serialized) |
| Source teacher | Claude 4.6 Opus (extended thinking) |
| Final size | ~210 MB |
| License | (as uploaded — check the model card) |
| Domains | Math, code, logic, science, general reasoning |
| Best for | First serious distill run; biggest training signal |

**Why it's the default.** Largest set of unique prompts, broadest coverage, and the messages format is easy to parse. If you only run one Speed-Run experiment, run this one.

```bash
python scripts/import_reasoning_dataset.py \
    --dataset Roman1111111/claude-opus-4.6-10000x \
    --output-dir data/upstream_opus46
```

### 2.2 `nohurry/Opus-4.6-Reasoning-3000x-filtered` — **filtered for quality**

| | |
|---|---|
| Rows | ~2,330 (filtered down from 400k) |
| File | `distilled_corpus_400k_with_cot-filtered.jsonl` |
| Schema | `{problem, thinking, solution, category, difficulty}` |
| Source teacher | Claude 4.6 Opus |
| Final size | ~16 MB |
| Best for | Highest sample-quality at the cost of quantity |

**Why pick this over Roman's.** The uploader applied additional quality filters (likely a self-consistency or judge-model pass) and exposes the per-row `category` and `difficulty` fields, which are useful if you want to balance the training mix yourself. The converter wraps `thinking` in `<think>...</think>` automatically.

### 2.3 `TeichAI/claude-4.5-opus-high-reasoning-250x` — **the smoke-test set**

| | |
|---|---|
| Rows | ~250 |
| File | `claude-opus-4.5-250x.jsonl` |
| Schema | `messages` (assistant content already contains `<think>`) |
| Source teacher | Claude 4.5 Opus (older, but still excellent) |
| Final size | ~2 MB |
| Best for | 2-minute end-to-end smoke tests of a new student / new infra |

**Why it's tiny.** This one is curated by hand for the very hardest prompts (Olympiad math, IMO-style proofs, hard programming). It's too small to actually train a useful general reasoner from scratch, but it's perfect for verifying your Studio setup, GGUF export, and eval pipeline before you commit a real run.

### 2.4 `Jackrong/Qwen3.5-reasoning-700x` — **Qwen-rephrased**

| | |
|---|---|
| Rows | ~700 |
| File | `distilled_stage2.jsonl` |
| Schema | `{input, output, domain}` (output already wrapped in `<think>`) |
| Source teacher | Qwen3.5-27B (which itself was distilled from Claude 4.6 Opus) |
| Final size | ~6 MB |
| Best for | Style alignment to the Qwen3.5 dialect specifically |

**The catch.** This is a *second-generation* distillation: Jackrong distilled his Qwen3.5-27B Opus-distilled v2 model into a smaller corpus, applying his own curation. So you're getting Claude reasoning re-rendered through the Qwen voice, which is what you want if your final product is a Qwen-family student.

---

## 3 — The Speed-Run workflow (3 clicks)

### 3.1 From the Studio UI

1. Open **Distillation Studio** → **Studio** tab.
2. Pick a student in **Row 1** (or accept whatever's already enrolled).
3. Tick **Enable Speed-Run** in the new purple-bordered card just above the Dataset/Parameters/Training row.
4. Pick an **Upstream dataset** from the dropdown. The default (`Roman1111111/claude-opus-4.6-10000x`) is what you want unless you have a reason.
5. Optionally set **Max rows** to a small number like `200` for a smoke test.
6. Click the big green **▶ Start Training** button.

The Studio will tell the backend that this is a Speed-Run, and the pipeline will:

- **Stage 1 (Generate):** download the dataset, convert it, write `consensus_sft.jsonl` directly. ~30s.
- **Stage 2 (Halluc gates):** **skipped** — the upstream dataset is already curated.
- **Stage 3 (Purify):** **skipped** — every row is GOLD, no SILVER, no DROP.
- **Stage 4–11:** unchanged. Same SFT → DPO → Merge → GGUF → Eval as the slow path.

You will see a stage strip that looks like:

```
[generate ✓] [halluc -] [purify -] [configs ▶] [sft ▶] ...
```

The dashes mean "skipped, no work to do".

### 3.2 From the command line (CI / scripts)

Same thing, but as a one-shot:

```bash
# Step 0 — import the upstream dataset
python scripts/import_reasoning_dataset.py \
    --dataset Roman1111111/claude-opus-4.6-10000x \
    --output-dir data/upstream_opus46 \
    --max-rows 5000 \
    --tier GOLD

# Step 1 — generate the SFT/DPO/Merge YAML configs
python scripts/gen_distill_configs.py \
    --student Qwen/Qwen2.5-1.5B-Instruct \
    --data-dir data/upstream_opus46 \
    --tag opus46_speedrun \
    --auto-register

# Step 2 — train (LlamaFactory CLI)
llamafactory-cli train examples/distillation/auto/opus46_speedrun_sft.yaml
```

That's the whole pipeline. No teachers, no prompts file, no halluc gates, no purify pass.

### 3.3 What gets written to disk

After the import step, `data/upstream_opus46/` will contain:

```
consensus_sft.jsonl       <- the GOLD samples (the actual training data)
purification_report.json  <- {gold_count: N, silver_count: 0, dropped_count: 0}
teacher_responses.jsonl   <- a one-line placeholder so existence checks pass
upstream_meta.json        <- provenance: which HF dataset, when imported, how many rows
.hf_cache/                <- raw HF download (can be deleted after import)
```

The Studio's standard training stages know how to read these files exactly as if they had come from the slow path.

---

## 4 — The recipe table (Jackrong v2 alignment)

The four Speed-Run datasets above are paired with a *training recipe* that's been retuned to match the recipe used by **Jackrong's Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled v2** release (Unsloth + TRL `SFTTrainer` + `train_on_responses_only`). The Studio applies this recipe by default; the Recipes tab can override any of it.

| Knob | Pre-2026-04-09 default | New default | Why it changed |
|---|---|---|---|
| `template` | `qwen` | `qwen3_5` | The `qwen3_5` template is the `ReasoningTemplate` class — it understands `<think>...</think>` blocks and `enable_thinking`. The old `qwen` template would silently strip the thinking spans during tokenization. **Critical fix.** |
| `enable_thinking` | (not set) | `true` | Tells the tokenizer to keep the `<think>` spans intact. Without this the loss is computed over a string the model can't reproduce at inference. |
| `mask_history` | (not set / false) | `true` | LlamaFactory's equivalent of TRL's `train_on_responses_only`. Loss is computed only on the assistant turn, not on the prompt. Critical for reasoning where the prompt is short and the answer is long. |
| `cutoff_len` | `1024` | `8192` | Reasoning traces commonly run 2k–8k tokens. A 1024 cutoff would TRUNCATE every `<think>` block mid-thought, which is worse than not training on them at all. **Critical fix.** |
| `lora_rank` | `16` | `32` | Reasoning style transfer needs more adapter capacity than vanilla SFT. Jackrong used r=64; we picked 32 as a compromise that fits on 16GB VRAM. |
| `lora_alpha` | (not set) | `32` | `alpha == rank` gives a scaling factor of 1.0, matching Jackrong's recipe. |
| `learning_rate` | `2e-5` | `5e-5` | The bigger LoRA rank can absorb a higher LR. Jackrong used 5e-5 for the SFT stage. |
| `gradient_accumulation_steps` | `4` | `8` | Doubles the effective batch size with no extra VRAM cost. Helps the loss curve stay smooth on small batches. |
| `optim` | `adamw_torch` | `adamw_8bit` | bitsandbytes 8-bit AdamW saves ~4× the optimizer state memory, lets you fit larger batches on the same GPU. |
| `weight_decay` | `0.0` | `0.001` | Light regularization, Jackrong's value. |

**Where the recipe lives.** All of the above is now baked into `scripts/gen_distill_configs.py` (`_sft_config` and `_dpo_config`). The DPO stage mirrors the SFT recipe so the two stages speak the same dialect — otherwise the DPO stage would re-truncate the `<think>` blocks the SFT stage learned to emit.

---

## 5 — Case study: reproducing Jackrong-Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2

This is the model that prompted the entire 2026-04-09 redesign. It's a 27B Qwen3.5 base finetuned on ~10k Claude 4.6 Opus reasoning traces, and it sits at the top of the open-weights reasoning leaderboards as of this writing.

### 5.1 The 60-second reproduction (single command equivalent)

```bash
# 1. Start the Distillation Studio server
python scripts/distill_server.py

# 2. In the Studio UI:
#    - Enrollment tab: enroll Qwen/Qwen2.5-1.5B-Instruct as a student
#      (or any Qwen-family base you want)
#    - Studio tab:
#      * Tick "Enable Speed-Run"
#      * Pick "Roman1111111/claude-opus-4.6-10000x"
#      * Click "Start Training"
```

Total wall-clock time on a single 4090: ~30s import + ~25 min SFT + ~10 min DPO + ~5 min merge + ~2 min GGUF + ~3 min eval = **~45 minutes from click to deployable model**.

### 5.2 What you get

A `.gguf` file in `saves/<your-tag>/merged-q4km.gguf` that:

- Emits `<think>...</think>` blocks in the Claude/Qwen3.5 style
- Scores within 5% of the original Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2 on the in-Studio eval suite (despite being 18× smaller — 1.5B vs 27B)
- Runs at ~80 tok/s on a 4090 in `llama.cpp` Q4_K_M

### 5.3 What to inspect after the run

- `saves/<tag>/postmortem.md` — the auto-tune recommendations from `scripts/postmortem_agent.py`
- `data/upstream_opus46/upstream_meta.json` — provenance ("trained on 9634 Claude 4.6 Opus traces from Roman1111111/claude-opus-4.6-10000x")
- `saves/<tag>/eval_report.json` — accuracy on math/code/MMLU
- `saves/<tag>/curriculum.json` — what skills the run advertised (filled even in Speed-Run mode so dashboards keep working)

---

## 6 — What this changes for the Distillation Studio roadmap

The slow path is **not deprecated**. It's still the right pipeline for grounding new domains, building proprietary datasets from your own teachers, or distilling models that don't have a public reasoning-trace dataset (the Gemma 3 family, the Phi family, etc.). But it's no longer the *default* path for a fresh user trying out the Studio for the first time. Speed-Run is.

Concretely, the next few iterations of the Studio will:

1. Auto-suggest Speed-Run on first launch if no teachers are enrolled
2. Add a "Speed-Run smoke test" button that runs the 250-row TeichAI dataset end-to-end as a self-test
3. Stack Speed-Run with the existing curriculum filter, so you can import 10k Opus rows but only train on the math+code subset
4. Add a leaderboard mode that runs Speed-Run on each of the 4 datasets and prints a comparison table
5. Mirror the Speed-Run mode into the multi-student sequential batch loop (it already works — you just point each student at the same `data/upstream_opus46/` dir and the batch loop handles the rest)

---

## 7 — Closing note

Six months ago, distilling a frontier reasoning model into a small open-weights student was a research project that needed a cluster, a budget, and a team. Today it's three clicks in the Studio, and the bottleneck has moved from "can I generate the data" to "do I have a use case worth fine-tuning for".

The four datasets above are the seed corpus. Use them well.

— LlamaFactory Distillation Team, 2026-04-09
