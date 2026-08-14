import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def compute_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def compute_precision(y_true, y_pred):
    return precision_score(y_true, y_pred)

def compute_recall(y_true, y_pred):
    return recall_score(y_true, y_pred)

def compute_f1(y_true, y_pred):
    return f1_score(y_true, y_pred)
    
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=1)

    return {
        "accuracy": compute_accuracy(labels, predictions),
        "precision": compute_precision(labels, predictions),
        "recall": compute_recall(labels, predictions),
        "f1": compute_f1(labels, predictions),
    }