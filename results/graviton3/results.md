# AWS Graviton3 Results

## Environment

- Instance: AWS EC2 c7g.xlarge
- Processor: AWS Graviton3
- Architecture: Arm64 (aarch64)
- vCPUs: 4
- Sequence length: 64
- Batch size: 1
- ONNX Runtime Execution Provider: CPUExecutionProvider

## Performance Results

| Stage | Accuracy | F1 | Avg Latency (ms) | Median (ms) | P95 (ms) | P99 (ms) | Throughput (samples/s) | Model Size (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PyTorch FP32 | — | — | 82.34 | 82.31 | 82.79 | 83.06 | 12.15 | — |
| ONNX FP32 | 91.17% | 91.09% | 75.56 | 75.52 | 75.93 | 76.09 | 13.23 | 450.11 |
| ONNX FP32 + Graph Opt (ALL) | — | — | 75.78 | — | — | — | — | 450.11 |
| ONNX FP32 + Graph Opt + Thread Tuning | — | — | 75.87 | 75.77 | — | — | 13.18 | 450.11 |
| ONNX INT8 + Graph Opt + Thread Tuning | 89.17% | 89.53% | 21.31 | 21.29 | — | — | 46.94 | 113.39 |

## Final Improvement

Compared with the PyTorch FP32 baseline:

- Average latency: 82.34 ms → 21.31 ms
- Latency reduction: 74.1%
- Inference speedup: 3.86x
- Throughput: 12.15 → 46.94 samples/sec
- Throughput improvement: 3.86x
- Model size: 450.11 MB → 113.39 MB
- Model size reduction: 74.81%
- Model compression: 3.97x
- Accuracy change: 91.17% → 89.17% (-2.00 percentage points)
- F1 change: 91.09% → 89.53% (-1.56 percentage points)