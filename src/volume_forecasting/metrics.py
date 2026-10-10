from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def wape(y_true, y_pred) -> float:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    denominator = np.abs(actual).sum()
    return np.nan if denominator == 0 else float(np.abs(actual - predicted).sum() / denominator)


def bias(y_true, y_pred) -> float:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    denominator = actual.sum()
    return np.nan if denominator == 0 else float((predicted - actual).sum() / denominator)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "wape": wape(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "bias": bias(y_true, y_pred),
    }
