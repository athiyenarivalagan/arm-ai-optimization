import numpy as np
import onnxruntime as ort

from src.evaluation.metrics import (
    compute_accuracy,
    compute_precision,
    compute_recall,
    compute_f1,
)

def evaluate_onnx(
    model_path,
    tokenizer,
    texts,
    labels,
    max_length,
):

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )
    
    predictions = []
    
    for text in texts:
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
            
        logits = session.run(None, onnx_inputs)[0]
        prediction = np.argmax(logits, axis=1)[0]
        predictions.append(prediction)

    predictions = np.asarray(predictions)

    return {
        "accuracy": compute_accuracy(labels, predictions),
        "precision": compute_precision(labels, predictions),
        "recall": compute_recall(labels, predictions),
        "f1": compute_f1(labels, predictions),
    }