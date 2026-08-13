"""
Generic utilities for inference benchmarking.
"""

import statistics
import time


def benchmark(inference_fn, warmup_runs=10, benchmark_runs=100):
    """
    Benchmark an inference function.

    Parameters
    ----------
    inference_fn : callable
        Function that performs one inference.

    warmup_runs : int
        Number of warmup iterations.

    benchmark_runs : int
        Number of measured benchmark iterations.

    Returns
    -------
    list[float]
        Per-inference latencies in milliseconds.
    """

    # Warmup
    for _ in range(warmup_runs):
        inference_fn()

    # Benchmark
    latencies = []

    for _ in range(benchmark_runs):
        start = time.perf_counter()

        inference_fn()

        end = time.perf_counter()

        latencies.append((end - start) * 1000)

    return latencies


def compute_latency_metrics(latencies):
    """
    Compute latency statistics from benchmark results.
    """

    if not latencies:
        raise ValueError("latencies must not be empty")

    sorted_latencies = sorted(latencies)

    return {
        "runs": len(latencies),
        "average": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "p95": sorted_latencies[int(len(latencies) * 0.95) - 1],
        "p99": sorted_latencies[int(len(latencies) * 0.99) - 1],
        "minimum": min(latencies),
        "maximum": max(latencies),
    }


def compute_throughput(avg_latency):
    """
    Compute throughput in samples/sec.

    Assumes batch size = 1.
    """

    if avg_latency <= 0:
        raise ValueError("avg_latency must be greater than zero")

    return 1000 / avg_latency


def print_benchmark_results(title, metrics):
    """
    Print benchmark results.
    """

    print("=" * 50)
    print(title)
    print("=" * 50)

    print(f"Runs               : {metrics['runs']}")
    print(f"Average latency    : {metrics['average']:.2f} ms")
    print(f"Median latency     : {metrics['median']:.2f} ms")
    print(f"P95 latency        : {metrics['p95']:.2f} ms")
    print(f"P99 latency        : {metrics['p99']:.2f} ms")
    print(f"Minimum latency    : {metrics['minimum']:.2f} ms")
    print(f"Maximum latency    : {metrics['maximum']:.2f} ms")
    print(f"Throughput         : {metrics['throughput']:.2f} samples/sec")

    print("=" * 50)