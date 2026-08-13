import onnxruntime as ort

from src.utils.benchmark_utils import (
    benchmark,
    compute_latency_metrics,
)


def create_session(model_path, optimization_level):
    """
    Create an ONNX Runtime session with the specified
    graph optimization level.
    """

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = optimization_level

    return ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )


def benchmark_optimization_levels(
    model_path,
    tokenizer,
    text,
    warmup_runs=10,
    benchmark_runs=100,
    max_length=64,
):
    """
    Compare ONNX Runtime graph optimization levels.
    """

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

    optimization_levels = {
        "Disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "Basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "Extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "All": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }

    results = {}

    for name, level in optimization_levels.items():

        session = create_session(
            model_path=model_path,
            optimization_level=level,
        )

        def run_inference():
            session.run(None, onnx_inputs)

        latencies = benchmark(
            run_inference,
            warmup_runs=warmup_runs,
            benchmark_runs=benchmark_runs,
        )

        metrics = compute_latency_metrics(latencies)

        results[name] = metrics

        print(
            f"{name:10} : "
            f"{metrics['average']:.2f} ms"
        )

    min_value = float("inf")
    target = ""
    
    for name, metrics in results.items():
        if metrics["average"] < min_value:
            min_value = metrics["average"]
            target = name  

    return results, optimization_levels[target]