import onnxruntime as ort

from src.utils.benchmark_utils import (
    benchmark,
    compute_latency_metrics,
    compute_throughput,
)


def create_session(
    model_path,
    intra_threads,
    inter_threads=1,
    optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
):
    """
    Create an ONNX Runtime session with a specified
    CPU thread configuration.
    """

    session_options = ort.SessionOptions()

    session_options.graph_optimization_level = optimization_level
    session_options.intra_op_num_threads = intra_threads
    session_options.inter_op_num_threads = inter_threads

    return ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )


def tune_threads(
    model_path,
    tokenizer,
    text,
    thread_counts=(1, 2, 4),
    inter_threads=1,
    warmup_runs=10,
    benchmark_runs=100,
    max_length=64,
    optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
):
    """
    Benchmark different ONNX Runtime intra-op thread counts.
    """

    # Prepare input
    inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    onnx_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }

    results = []

    # Test each thread configuration
    for intra_threads in thread_counts:

        session = create_session(
            model_path=model_path,
            intra_threads=intra_threads,
            inter_threads=inter_threads,
            optimization_level=optimization_level,
        )

        def run_inference():
            session.run(None, onnx_inputs)

        latencies = benchmark(
            run_inference,
            warmup_runs=warmup_runs,
            benchmark_runs=benchmark_runs,
        )

        metrics = compute_latency_metrics(latencies)
        metrics["throughput"] = compute_throughput(
            metrics["average"]
        )

        results.append(
            {
                "intra_threads": intra_threads,
                "inter_threads": inter_threads,
                **metrics,
            }
        )

    # Select configuration with lowest median latency
    best_result = min(
        results,
        key=lambda result: result["median"],
    )

    return results, best_result