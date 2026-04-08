# Distillation Config & Output Schemas

Reference for all configuration and output formats used by the distillation pipeline.

---

## Teacher Manifest (`manifest.json`)

```json
{
  "teachers": [
    {
      "name": "gemma4-26b",
      "model_path": "/path/to/gemma-4-26b-it-Q4_K_M.gguf",
      "context_length": 8192,
      "weight": 1.5
    }
  ]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Short identifier for the teacher |
| `model_path` | string | yes | Absolute path to the GGUF file |
| `context_length` | int | no | Max context window (default: 8192) |
| `weight` | float | no | Voting weight for purification (default: 1.0) |

---

## Prompts (`prompts.jsonl`)

```jsonl
{"id": "med-001", "prompt": "Explain the mechanism of action of metformin.", "category": "medical"}
{"id": "code-001", "prompt": "Write a Python function to merge two sorted lists.", "category": "coding"}
```

| Field | Type | Required |
|-------|------|----------|
| `id` | string | yes |
| `prompt` | string | yes |
| `category` | string | no |

---

## Teacher Responses (`teacher_responses.jsonl`)

```jsonl
{"prompt_id": "med-001", "teacher": "gemma4-26b", "response": "Metformin works by...", "tokens": 245, "elapsed_s": 3.2}
```

---

## Purified Output — GOLD (`consensus_sft.jsonl`)

```jsonl
{"instruction": "Explain the mechanism of action of metformin.", "output": "Metformin works by...", "source_teachers": ["gemma4-26b", "qwen-14b"], "confidence": 0.95}
```

---

## Purified Output — SILVER (`conflict_dpo.jsonl`)

```jsonl
{"instruction": "...", "chosen": "Best answer...", "rejected": "Weaker answer...", "chosen_teacher": "gemma4-26b", "rejected_teacher": "mistral-7b"}
```

---

## Forge Matrix (`<tag>_matrix.yaml`)

```yaml
tag: zena007
sft_data: data/zena007/purified/consensus_sft.jsonl
dpo_data: data/zena007/purified/conflict_dpo.jsonl
max_parallel: 2
eval_probe_split: 0.1

variants:
  - name: qwen-r16
    model: Qwen/Qwen2.5-1.5B-Instruct
    lora_rank: 16
    learning_rate: 2.0e-4
    num_train_epochs: 3

  - name: qwen-r32
    model: Qwen/Qwen2.5-1.5B-Instruct
    lora_rank: 32
    learning_rate: 1.0e-4
    num_train_epochs: 5
```

---

## SFT Training Config (auto-generated YAML)

```yaml
model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 16
dataset: zena007_consensus_sft
output_dir: saves/zena007/sft
overwrite_output_dir: false
per_device_train_batch_size: 4
learning_rate: 2.0e-4
num_train_epochs: 3
bf16: true
```

---

## Graduation Report (`graduation_report.json`)

```json
{
  "tag": "zena007",
  "verdict": "PASS",
  "overall_retention": 0.87,
  "categories": {
    "medical": {"retention": 0.91, "confidence_low": 0.85, "confidence_high": 0.96, "status": "pass"},
    "coding": {"retention": 0.82, "confidence_low": 0.74, "confidence_high": 0.89, "status": "pass"}
  },
  "emergences": [],
  "ruins": [],
  "champion": "qwen-r16",
  "timestamp": "2026-04-07T12:00:00Z"
}
```
