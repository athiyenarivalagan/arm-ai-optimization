import torch

from torch.profiler import (
    profile,
    record_function,
    ProfilerActivity,
)


def profile_pytorch(
    model,
    tokenizer,
    text,
    warmup_runs=10,
    max_length=64,
    row_limit=20,
):
    """
    Profile PyTorch CPU inference for a single input sample.
    """

    # Evaluation mode
    model.eval()

    # Prepare input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )

    # Warmup (Is it necessary to warn-up before running the profiler?)
    for _ in range(warmup_runs):
        with torch.inference_mode():
            model(**inputs)

    # Profile
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
    ) as prof:

        with record_function("ERNIE Inference"):
            with torch.inference_mode():
                model(**inputs)

    # Display results
    print(
        prof.key_averages().table(
            sort_by="cpu_time_total",
            row_limit=row_limit,
        )
    )