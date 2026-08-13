import torch, onnx

def export_onnx(
    model,
    tokenizer,
    text,
    output_path,
    max_length=64,
    opset_version=17,
):

    """
    Export a PyTorch sequence classification model to ONNX.
    """

    model.eval()

    # Prepare example inputs for ONNX tracing/export
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )

    # Representative example tensors that tell the exporter what the model's inputs look like
    dummy_inputs = (
        inputs["input_ids"],
        inputs["attention_mask"],
    )

    torch.onnx.export(
        model,
        dummy_inputs,
        output_path,
        input_names=[
            "input_ids",
            "attention_mask",
        ],
        output_names=[
            "logits",
        ],
        dynamic_axes={
            "input_ids": {
                0: "batch_size",
                1: "sequence_length",
            },
            "attention_mask": {
                0: "batch_size",
                1: "sequence_length",
            },
            "logits": {
                0: "batch_size",
            },
        },
        opset_version=opset_version,
    )

    print(f"ONNX model exported to: {output_path}")

def validate_onnx(model_path):
    """
    Validate an exported ONNX model.
    """

    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    print("ONNX model is valid!")