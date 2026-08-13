import onnxruntime as ort

from src.utils.benchmark_utils import (
    benchmark,
    compute_latency_metrics,
    compute_throughput,
    print_benchmark_results,
)


def benchmark_onnx(
    model_path,
    tokenizer,
    text,
    warmup_runs=10,
    benchmark_runs=100,
    max_length=64,
):

    """
    Benchmark ONNX Runtime inference for a single input sample
    """

    # Create ONNX Runtime session 
    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )

    
    # TEMPORARY: inspect ONNX Runtime configuration
    print("Providers:", session.get_providers())

    options = session.get_session_options()

    print("Graph optimization:", options.graph_optimization_level)
    print("Intra-op threads:", options.intra_op_num_threads)
    print("Inter-op threads:", options.inter_op_num_threads)
    print("Execution mode:", options.execution_mode)

    
    inputs = tokenizer(
        text,
        # return_tensors="pt",
        return_tensors="np", # ONNX Runtime expects NumPy arrays
        truncation=True,
        padding=True,
        max_length=max_length,
    )

    print("Input shape:", inputs["input_ids"].shape) # code for debugging 
    
    onnx_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }
    
    # onnx_logits = session.run(None, onnx_inputs)[0]

    # ONNX inference
    def run_inference():
        session.run(None, onnx_inputs)

    # Benchmark
    latencies = benchmark(
        run_inference,
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
    )

    # Compute metrics
    metrics = compute_latency_metrics(latencies)
    metrics["throughput"] = compute_throughput(
        metrics["average"]
    )

    # Display results
    print_benchmark_results(
        "ONNX Runtime Benchmark",
        metrics,
    )

    return metrics