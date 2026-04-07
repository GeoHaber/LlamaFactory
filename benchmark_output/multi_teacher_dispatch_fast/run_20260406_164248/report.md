# Multi-Teacher Dispatch Benchmark

- Timestamp: 20260406_164248
- Prompt samples: 2
- Teachers: 2

## Results

| Mode | Exit | Wall (s) | New | Resumed | Skipped | Teacher Calls | Teacher Elapsed (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| teacher-sequential | 0 | 154.22 | 4 | 0 | 0 | 4 | 145.49 |
| teacher-fifo | 0 | 116.4 | 4 | 0 | 0 | 4 | 162.68 |

- Speedup (sequential/fifo): 1.325x

## Artifacts

- JSON report: benchmark_output\multi_teacher_dispatch_fast\run_20260406_164248\report.json
- Raw logs: benchmark_output\multi_teacher_dispatch_fast\run_20260406_164248
