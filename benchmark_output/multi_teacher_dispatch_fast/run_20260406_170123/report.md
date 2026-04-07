# Multi-Teacher Dispatch Benchmark

- Timestamp: 20260406_170123
- Prompt samples: 1
- Teachers: 2

## Results

| Mode | Exit | Wall (s) | New | Resumed | Skipped | Teacher Calls | Teacher Elapsed (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| teacher-sequential | 0 | 87.72 | 2 | 0 | 0 | 2 | 83.78 |
| teacher-fifo | 0 | 77.5 | 2 | 0 | 0 | 2 | 72.76 |

- Speedup (sequential/fifo): 1.132x

## Artifacts

- JSON report: benchmark_output\multi_teacher_dispatch_fast\run_20260406_170123\report.json
- Raw logs: benchmark_output\multi_teacher_dispatch_fast\run_20260406_170123
