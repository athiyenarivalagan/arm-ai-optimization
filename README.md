## ERNIE Inference Optimization on Arm

Optimized ERNIE sentiment-classification inference for AWS Graviton3 using ONNX Runtime, graph optimization, thread tuning, 
and dynamic INT8 quantization.

## Project Overview

Transformer-based language models can be computationally expensive to deploy on CPU infrastructure. 
This project explores how an ERNIE model for Chinese sentiment classification can be optimized for efficient inference on Arm-based
servers.

Starting from a fine-tuned PyTorch FP32 model, the inference pipeline is exported to ONNX and optimized using ONNX Runtime. 
Graph optimization, CPU thread tuning, and dynamic INT8 quantization are evaluated on an AWS Graviton3 instance.

The final INT8 model reduces average inference latency from 82.34 ms to 21.31 ms and model size from 450.11 MB to 113.39 MB, 
while limiting the accuracy reduction to 2.00 percentage points.

## Why this project / why it should win


This project demonstrates a practical end-to-end workflow for deploying transformer inference efficiently on Arm server CPUs.

Rather than evaluating optimization techniques in isolation, the project measures their effect on latency, throughput, model size, and prediction quality on real AWS Graviton3 hardware.

The final optimized model achieves:
```
- 3.86x faster inference compared with the PyTorch FP32 baseline
- 3.86x higher throughput
- 74.81% smaller model size (3.97x reduction)
- 89.17% accuracy, compared with 91.17% for the FP32 model
```
The project also reports optimization attempts that did not improve performance. Graph optimization and thread tuning were benchmarked on the target hardware but produced little additional improvement for FP32 inference, while INT8 quantization delivered the largest performance gain.

This provides a reproducible example of measuring optimization trade-offs rather than assuming that every optimization improves performance.


## Optimization pipeline
```
  Fine-tuned ERNIE FP32
            │
            ▼
         PyTorch
            │
            ▼
       ONNX Export
            │
            ▼
    ONNX Runtime FP32
            │
            ├── Graph Optimization
            │
            ├── Thread Tuning
            │
            ▼
  Dynamic INT8 Quantization
            │
            ▼
    ONNX Runtime INT8
            │
            ▼
      AWS Graviton3
            │
            ▼
Benchmark + Quality Evaluation
```

## Results 

**Final result:** 3.86x faster inference and a 3.97x smaller model on AWS Graviton3, with a 2.00 percentage-point accuracy trade-off.

### Benchmark Environment

- Platform: AWS Graviton3
- EC2 instance: `c7g.xlarge`
- Architecture: Arm64 (`aarch64`)
- vCPUs: 4
- Sequence length: 64
- Batch size: 1
- Warmup runs: 10
- Benchmark runs: 100
- ONNX Runtime provider: `CPUExecutionProvider`

### Performance

| Model | Accuracy | F1 | Avg Latency | Throughput | Model Size |
|---|---:|---:|---:|---:|---:|
| PyTorch FP32 | — | — | 82.34 ms | 12.15 samples/s | — |
| ONNX FP32 | 91.17% | 91.09% | 75.56 ms | 13.23 samples/s | 450.11 MB |
| ONNX FP32 + Graph Optimization | — | — | 75.78 ms | — | 450.11 MB |
| ONNX FP32 + Graph Optimization + Thread Tuning | — | — | 75.87 ms | 13.18 samples/s | 450.11 MB |
| **ONNX INT8 + Graph Optimization + Thread Tuning** | **89.17%** | **89.53%** | **21.31 ms** | **46.94 samples/s** | **113.39 MB** |

### Overall Improvement

Compared with the PyTorch FP32 baseline:

- Average latency: **82.34 ms → 21.31 ms (74.1% reduction, 3.86x speedup)**
- Throughput: **12.15 → 46.94 samples/s (3.86x)**
- Model size: **450.11 MB → 113.39 MB (74.81% reduction, 3.97x smaller)**
- Accuracy: **91.17% → 89.17% (-2.00 percentage points)**
- F1: **91.09% → 89.53% (-1.56 percentage points)**

## Functionality/output


## Project structure 

```
arm-ai-optimization/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── data/
│   └── ChnSentiCorp/
│
├── models/
│
├── notebooks/
│   └── ernie.ipynb
│
├── results/
│   └── graviton3/
│       └── results.md
│
└── src/
    ├── benchmark/
    │   ├── benchmark_pytorch.py
    │   ├── benchmark_onnx.py
    │   └── benchmark_int8.py
    │
    ├── evaluation/
    │   ├── evaluate_onnx.py
    │   └── metrics.py
    │
    ├── optimization/
    │   ├── export_onnx.py
    │   ├── optimize_onnx.py
    │   ├── quantize.py
    │   └── tune_threads.py
    │
    ├── profiling/
    │   ├── profile_pytorch.py
    │   └── profile_onnx.py
    │
    └── utils/
        ├── benchmark_utils.py
        └── load_dataset.py
```

## Arm64 setup 

```
git clone ...
cd arm-ai-optimization

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

```
1. Launch an Arm64 AWS Graviton instance.
2. Install Python.
3. Clone repository.
4. Install requirements.
5. Run/export model.
6. Run FP32 benchmark.
7. Quantize model.
8. Run INT8 benchmark.
9. Compare results.
```

## Reproducing results 
