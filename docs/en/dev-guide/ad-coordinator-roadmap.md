# AD Coordinator Integration Roadmap

This roadmap operationalizes a multi-role local inference pipeline for:
- Python code review
- Python-to-Rust translation
- Distillation to smaller local students

## Phase 1: Coordinator Core

Status: In progress (implemented initial vertical slice)

Goals:
- Introduce an adaptive coordinator with role-based execution.
- Keep existing single-model flow backward compatible.

Delivered:
- Added `ADCoordinator` with `fast`, `balanced`, and `quality` policies.
- Integrated coordinator into `ChatModel` as an optional inference mode.
- Added complexity-based routing heuristic.

Current constraints:
- First version reuses the same loaded model for planner/coder/logician roles.
- Streaming in coordinator mode yields final aggregated output (not token-level role streaming yet).

## Phase 2: UI and Local Runtime Integration

Status: In progress (controls added)

Goals:
- Enable coordinator mode directly from Chat tab.
- Expose policy and complexity threshold in UI.

Delivered:
- Added Chat tab controls:
  - enable coordinator
  - coordinator policy
  - complexity threshold
- Added `ktransformers` backend to match supported engines.
- Added hardware auto-tune controls:
  - auto tune toggle
  - preferred policy target
  - runtime recommendation applied on model load

Next:
- Add role-specific model selectors and presets.
- Add coordinator trace panel (planner/coder/logician sections).

## Phase 3: API Coordinator Mode

Status: planned

Goals:
- Add optional coordinator fields in API requests.
- Add structured coordinator traces in API responses.
- Support streaming tool-calls in coordinator mode.

Next steps:
- Extend protocol models with coordinator options.
- Add runtime override guardrails to avoid global mutable state conflicts.

## Phase 4: Distillation Workflow Productization

Status: started (example configs scaffolded)

Goals:
- Turn teacher-cocktail into a reproducible pipeline.
- Produce small local students with strong coding/reasoning behavior.

Pipeline:
1. Generate teacher traces (planner/coder/logician outputs)
2. SFT distillation on curated traces
3. Preference optimization (DPO/KTO)
4. Quantization and serving benchmarks

## Phase 5: Testing and Benchmarking

Status: started

Goals:
- Validate coordinator routing correctness.
- Protect single-model behavior from regressions.

Delivered:
- Added coordinator unit tests for policy routing and refinement flow.

Next:
- Add API coordinator contract tests.
- Add quality/latency benchmark harness for Python-to-Rust tasks.

## Acceptance Criteria

- Coordinator mode can be enabled from UI without manual JSON hacks.
- Single-model mode remains unchanged and default.
- Distillation templates can be run with only model path and dataset substitutions.
- Runtime can auto-select backend/dtype/policy/quantization from machine resources.
- Benchmark report includes:
  - compile success rate
  - test pass rate
  - p50/p95 latency
  - token usage
