"""
metrics.py — Metric computation for each task type.
"""

from sklearn.metrics import accuracy_score, f1_score
import numpy as np


def classification_metrics(y_true: list, y_pred: list) -> dict:
    """Accuracy and macro-F1 for classification tasks."""
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": round(acc, 4), "macro_f1": round(f1, 4)}


def regression_mse(y_true: list[float], y_pred: list[float]) -> dict:
    """Mean Squared Error for sentiment scoring (FiQA-SA)."""
    mse = float(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))
    return {"mse": round(mse, 6)}


def exact_match(y_true: list[str], y_pred: list[str]) -> dict:
    """
    Numeric Exact Match for ConvFinQA.
    Strips trailing zeros and compares as strings after rounding to 4 dp.
    """
    def normalise(v: str) -> str:
        try:
            return str(round(float(v), 4))
        except (ValueError, TypeError):
            return str(v).strip().lower()

    correct = sum(normalise(t) == normalise(p) for t, p in zip(y_true, y_pred))
    em = correct / len(y_true) if y_true else 0.0
    return {"exact_match": round(em, 4), "correct": correct, "total": len(y_true)}


def compute_metrics(task_type: str, y_true: list, y_pred: list) -> dict:
    """Dispatch to the correct metric function."""
    if task_type == "classification":
        return classification_metrics(y_true, y_pred)
    elif task_type == "regression":
        return regression_mse(y_true, y_pred)
    elif task_type == "exact_match":
        return exact_match(y_true, y_pred)
    else:
        raise ValueError(f"Unknown task type: {task_type}")
