import torch

from src.utils.benchmark_utils import (
    benchmark,
    compute_latency_metrics,
    compute_throughput,
    print_benchmark_results,
)


def benchmark_pytorch(
    model,
    tokenizer,
    text,
    warmup_runs=10,
    benchmark_runs=100,
    max_length=64,
):
    """
    Benchmark PyTorch inference for a single input sample.
    """

    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )

    # print("Input shape:", inputs["input_ids"].shape) # for debugging 

    def run_inference():
        with torch.inference_mode():
            model(**inputs)

    latencies = benchmark(
        run_inference,
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
    )

    metrics = compute_latency_metrics(latencies)
    metrics["throughput"] = compute_throughput(metrics["average"])

    print_benchmark_results(
        "PyTorch Benchmark",
        metrics,
    )

    return metrics