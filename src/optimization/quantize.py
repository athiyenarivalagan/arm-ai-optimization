from pathlib import Path
import onnx

from onnxruntime.quantization import (
    quantize_dynamic, 
    QuantType,
)

def quantize_model(
    input_model_path,
    relaxed_model_path,
    output_model_path,
):
    """
    Dynamically quantize an ONNX model to INT8 weights
    """

    # Load FP32 ONNX model
    model = onnx.load(
        input_model_path,
        load_external_data=True,
    )

    # Remove conflicting intermediate shape metadata.
    # This forces ONNX to calculate the shape dynamically instead of checking hardcoded shapes.
    model.graph.ClearField("value_info")
    
    # Save cleaned model
    onnx.save_model(
        model,
        relaxed_model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=Path(relaxed_model_path).name + ".data",
        size_threshold=1024,
    )

    # Dynamic INT8 quantization
    quantize_dynamic(
        model_input=relaxed_model_path,
        model_output=output_model_path,
        per_channel=True,
        weight_type=QuantType.QInt8,
        extra_options={
            "DisableShapeInference": True,
        },
    )
    
    # Validate generated model
    onnx.checker.check_model(output_model_path)

    print(f"INT8 model written to: {output_model_path}")