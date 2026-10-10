from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from .metrics import regression_metrics


@dataclass
class TrainingResult:
    model: Pipeline
    metrics: dict[str, float]
    predictions: pd.DataFrame
    metadata: dict


class ModelTrainer:
    """Trains the selected production model with a chronological holdout."""

    def __init__(self, features: list[str], alpha: float = 10.0, test_months: int = 12) -> None:
        self.features = features
        self.alpha = alpha
        self.test_months = test_months

    def _pipeline(self) -> Pipeline:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=self.alpha)),
        ])

    def fit_evaluate(self, data: pd.DataFrame) -> TrainingResult:
        required = ["date", "Volume", "Log_Volume", *self.features]
        clean = data[required].replace([np.inf, -np.inf], np.nan).dropna().sort_values("date")
        if len(clean) <= self.test_months:
            raise ValueError("Insufficient data for the requested holdout.")
        train = clean.iloc[:-self.test_months].copy()
        test = clean.iloc[-self.test_months:].copy()
        model = self._pipeline().fit(train[self.features], train["Log_Volume"])
        prediction = np.maximum(np.expm1(model.predict(test[self.features])), 0)
        metrics = regression_metrics(test["Volume"], prediction)
        predictions = pd.DataFrame({
            "date": test["date"].values,
            "actual_volume": test["Volume"].values,
            "predicted_volume": prediction,
        })
        metadata = {
            "model_name": "Ridge",
            "alpha": self.alpha,
            "target": "Log_Volume",
            "features": self.features,
            "test_months": self.test_months,
            "train_start": train["date"].min(),
            "train_end": train["date"].max(),
            "test_start": test["date"].min(),
            "test_end": test["date"].max(),
            "metrics": metrics,
        }
        return TrainingResult(model, metrics, predictions, metadata)

    def refit_all(self, data: pd.DataFrame) -> Pipeline:
        required = ["Log_Volume", *self.features]
        clean = data[required].replace([np.inf, -np.inf], np.nan).dropna()
        return self._pipeline().fit(clean[self.features], clean["Log_Volume"])
