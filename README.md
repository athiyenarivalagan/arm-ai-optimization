# ERNIE Inference Optimization on Arm

Optimized ERNIE sentiment-classification inference for AWS Graviton3 using ONNX Runtime, graph optimization, thread tuning, and dynamic INT8 quantization.

## Project Overview

Transformer-based language models can be computationally expensive to deploy on CPU infrastructure. This project explores how an ERNIE model for Chinese sentiment classification can be optimized for efficient inference on Arm-based servers.

Starting from a fine-tuned PyTorch FP32 model, the inference pipeline is exported to ONNX and optimized using ONNX Runtime. Graph optimization, CPU thread tuning, and dynamic INT8 quantization are evaluated on an AWS Graviton3 instance.

The final INT8 model reduces average inference latency from **82.34 ms to 21.31 ms** and model size from **450.11 MB to 113.39 MB**, while limiting the accuracy reduction to **2.00 percentage points**.

## Why This Project Matters

This project demonstrates a practical end-to-end workflow for deploying transformer inference efficiently on Arm server CPUs.

Rather than evaluating optimization techniques in isolation, the project measures their effect on latency, throughput, model size, and prediction quality on real AWS Graviton3 hardware.

The final optimized model achieves:

- **3.86x faster inference** compared with the PyTorch FP32 baseline
- **3.86x higher throughput**
- **74.81% smaller model size (3.97x reduction)**
- **89.17% accuracy**, compared with 91.17% for the FP32 model

The project also reports optimization attempts that did not improve performance. Graph optimization and thread tuning were benchmarked on the target hardware but produced little additional improvement for FP32 inference, while INT8 quantization delivered the largest performance gain.

This provides a reproducible example of measuring optimization trade-offs rather than assuming that every optimization improves performance.

## Optimization Pipeline

```text
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

Key improvements from the optimization pipeline:

- Average latency: **82.34 ms → 21.31 ms (74.1% reduction, 3.86x speedup)** compared with PyTorch FP32
- Throughput: **12.15 → 46.94 samples/s (3.86x)** compared with PyTorch FP32
- Model size: **450.11 MB → 113.39 MB (74.81% reduction, 3.97x smaller)** compared with ONNX FP32
- Accuracy: **91.17% → 89.17% (-2.00 percentage points)** compared with ONNX FP32
- F1: **91.09% → 89.53% (-1.56 percentage points)** compared with ONNX FP32

## Functionality and Output

The project provides an end-to-end workflow for optimizing and evaluating ERNIE inference on Arm CPUs.

It supports:

- Fine-tuning ERNIE for Chinese sentiment classification
- PyTorch FP32 inference benchmarking
- PyTorch-to-ONNX export
- ONNX Runtime FP32 inference
- ONNX graph optimization benchmarking
- CPU thread tuning
- Dynamic INT8 quantization
- FP32 and INT8 model-quality evaluation
- Latency, throughput, and model-size comparison
- CPU profiling utilities

The final output is an INT8 ONNX model optimized and benchmarked for CPU inference on AWS Graviton3.

Generated model binaries are not stored in the repository because of their size. They can be reproduced locally through the optimization pipeline.

## Project Structure

```text
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
    │   └── profile_pytorch.py
    │
    └── utils/
        ├── benchmark_utils.py
        └── load_dataset.py
```

## Arm64 Setup

The final benchmarks were collected on an AWS EC2 `c7g.xlarge` instance powered by AWS Graviton3.

### 1. Create an Arm64 Environment

Launch an Arm64 Linux environment. The reported results were produced using:

- AWS EC2 `c7g.xlarge`
- AWS Graviton3
- Arm64 / `aarch64`
- 4 vCPUs
- Ubuntu 24.04 LTS

Verify the architecture:

```bash
uname -m
```

Expected output:

```text
aarch64
```

### 2. Clone the Repository

```bash
git clone https://github.com/athiyenarivalagan/arm-ai-optimization.git
cd arm-ai-optimization
```

### 3. Create the Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Prepare ChnSentiCorp

The ChnSentiCorp dataset is third-party data and is not redistributed in this repository.

Obtain the dataset separately and place the files in:

```text
data/ChnSentiCorp/
├── train.tsv
├── dev.tsv
└── test.tsv
```

The dataset used for this project is loaded as:

- Training: 9,146 examples
- Development: 1,200 examples
- Test: 1,200 examples

Each TSV file uses the following format:

```text
label    text_a
1        ...
0        ...
```

### 5. Start Jupyter

```bash
jupyter lab
```

Open:

```text
notebooks/ernie.ipynb
```

The notebook contains the complete fine-tuning, export, optimization, quantization, evaluation, and benchmarking workflow.

## Reproducing Results

The optimization pipeline can be reproduced from `notebooks/ernie.ipynb`.

### Step 1 — Fine-tune ERNIE

The project starts from the pretrained model:

```text
nghuyong/ernie-3.0-base-zh
```

Set:

```python
RUN_FINETUNING = True
```

and execute the fine-tuning section of the notebook.

The fine-tuned model and tokenizer are saved to:

```text
models/ernie-finetuned/
```

Fine-tuning is stochastic, so exact quality metrics may vary slightly between runs unless deterministic seeds and identical software/hardware conditions are used.

The checkpoint used for the reported Graviton3 benchmarks achieved **91.17% accuracy** and **91.09% F1**.

### Step 2 — Benchmark PyTorch FP32

Run the PyTorch benchmark section to establish the unoptimized inference baseline.

Reported Graviton3 result:

```text
Average latency: 82.34 ms
Throughput:      12.15 samples/sec
```

### Step 3 — Export to ONNX

Export the fine-tuned PyTorch model to ONNX.

The generated model is written to:

```text
models/ernie_finetuned.onnx
```

Large generated model files are excluded from Git and are produced locally by the pipeline.

### Step 4 — Benchmark ONNX Runtime

Run the FP32 ONNX model using ONNX Runtime with `CPUExecutionProvider`.

Reported Graviton3 result:

```text
Average latency: 75.56 ms
Throughput:      13.23 samples/sec
```

### Step 5 — Evaluate Graph Optimization Levels

Benchmark the following ONNX Runtime graph optimization levels:

```text
ORT_DISABLE_ALL
ORT_ENABLE_BASIC
ORT_ENABLE_EXTENDED
ORT_ENABLE_ALL
```

The best level is selected using average latency on the target system.

On the reported Graviton3 run, `ORT_ENABLE_ALL` produced an average latency of **75.78 ms**.

### Step 6 — Tune CPU Threads

Benchmark different intra-op thread counts while keeping inter-op threads fixed.

The Graviton3 experiment evaluated:

```text
Intra-op: 1, 2, 4
Inter-op: 1
```

The selected configuration was:

```text
Intra-op threads: 4
Inter-op threads: 1
```

### Step 7 — Apply Dynamic INT8 Quantization

Quantize the ONNX model using ONNX Runtime dynamic quantization with signed INT8 weights.

The generated model is:

```text
models/ernie_finetuned_int8.onnx
```

Model size is reduced from:

```text
450.11 MB → 113.39 MB
```

representing a **74.81% reduction (3.97x smaller)**.

### Step 8 — Benchmark the Optimized INT8 Model

Apply the selected graph optimization and thread configuration to the INT8 model and run the final benchmark.

Reported Graviton3 result:

```text
Average latency: 21.31 ms
Median latency:  21.29 ms
Throughput:      46.94 samples/sec
```

### Step 9 — Validate Model Quality

Evaluate the FP32 and INT8 ONNX models on the same development dataset.

Reported results:

| Model | Accuracy | F1 |
|---|---:|---:|
| ONNX FP32 | 91.17% | 91.09% |
| ONNX INT8 | 89.17% | 89.53% |

Dynamic INT8 quantization therefore resulted in:

- Accuracy change: **-2.00 percentage points**
- F1 change: **-1.56 percentage points**
- Average latency reduction: **74.1%**
- Inference speedup: **3.86x**
- Model size reduction: **74.81%**

Fine-tuning is stochastic, so exact quality metrics may vary slightly when training a new checkpoint.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
