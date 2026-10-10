from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import make_scorer

from .metrics import wape, regression_metrics


@dataclass
class ExperimentOutput:
    results: pd.DataFrame
    predictions: pd.DataFrame
    fitted_models: dict[str, Any]
    best_result: pd.Series
    best_model: Any


class ForecastExperiment:
    """Runs leakage-safe model comparison on a common chronological holdout."""

    def __init__(self, feature_sets: dict[str, list[str]], test_months: int = 12,
                 max_abs_bias: float = 0.07, random_state: int = 42) -> None:
        self.feature_sets = feature_sets
        self.test_months = test_months
        self.max_abs_bias = max_abs_bias
        self.random_state = random_state

    @staticmethod
    def _linear_pipeline(model) -> Pipeline:
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler()), ("model", model)])

    def model_configs(self) -> dict[str, dict[str, Any]]:
        return {
            "LinearRegression": {"pipeline": self._linear_pipeline(LinearRegression()), "param_grid": {}},
            "Ridge": {"pipeline": self._linear_pipeline(Ridge()),
                      "param_grid": {"model__alpha": [0.1, 0.5, 1, 5, 10, 25, 50, 100]}},
            "ElasticNet": {"pipeline": self._linear_pipeline(ElasticNet(max_iter=20000, random_state=self.random_state)),
                           "param_grid": {"model__alpha": [0.01, 0.05, 0.1, 0.5, 1.0],
                                          "model__l1_ratio": [0.1, 0.25, 0.5, 0.75]}},
            "RandomForest": {"pipeline": Pipeline([("imputer", SimpleImputer(strategy="median")),
                                                     ("model", RandomForestRegressor(random_state=self.random_state, n_jobs=-1))]),
                             "param_grid": {"model__n_estimators": [100, 300], "model__max_depth": [2, 3],
                                            "model__min_samples_leaf": [3, 5], "model__max_features": [0.7, 1.0]}},
            "GradientBoosting": {"pipeline": Pipeline([("imputer", SimpleImputer(strategy="median")),
                                                        ("model", GradientBoostingRegressor(random_state=self.random_state))]),
                                 "param_grid": {"model__n_estimators": [20, 50, 100],
                                                "model__learning_rate": [0.01, 0.03, 0.05],
                                                "model__max_depth": [1, 2], "model__min_samples_leaf": [3, 5],
                                                "model__loss": ["squared_error", "huber"]}},
        }

    @staticmethod
    def _log_wape(y_true_log, y_pred_log) -> float:
        return wape(np.expm1(y_true_log), np.maximum(np.expm1(y_pred_log), 0))

    @staticmethod
    def _cv(n_train: int) -> TimeSeriesSplit:
        splits = 5 if n_train >= 24 else 3 if n_train >= 16 else 2
        return TimeSeriesSplit(n_splits=splits, test_size=1)

    def _test_dates(self, data: pd.DataFrame) -> pd.DatetimeIndex:
        dates = data["date"].dropna().sort_values().unique()
        if len(dates) <= self.test_months:
            raise ValueError("Insufficient observations for requested holdout.")
        return pd.DatetimeIndex(dates[-self.test_months:])

    @staticmethod
    def _split(data: pd.DataFrame, features: list[str], test_dates: pd.DatetimeIndex):
        columns = ["date", "Volume", "Log_Volume", *features]
        subset = data[columns].replace([np.inf, -np.inf], np.nan).dropna().sort_values("date")
        train = subset[subset["date"] < test_dates.min()].copy()
        test = subset[subset["date"].isin(test_dates)].copy()
        if len(test) != len(test_dates):
            raise ValueError("Feature set is incomplete for one or more holdout dates.")
        if len(train) < 10:
            raise ValueError("Feature set has fewer than ten training rows.")
        return train, test

    def run(self, data: pd.DataFrame) -> ExperimentOutput:
        scorer = make_scorer(self._log_wape, greater_is_better=False)
        test_dates = self._test_dates(data)
        rows, prediction_frames, fitted = [], [], {}
        configs = self.model_configs()
        for set_name, features in self.feature_sets.items():
            train, test = self._split(data, features, test_dates)
            for model_name, config in configs.items():
                pipeline = clone(config["pipeline"])
                if config["param_grid"]:
                    search = GridSearchCV(pipeline, config["param_grid"], scoring=scorer,
                                          cv=self._cv(len(train)), n_jobs=-1, refit=True, error_score="raise")
                    search.fit(train[features], train["Log_Volume"])
                    fitted_model, params, cv_wape = search.best_estimator_, search.best_params_, -search.best_score_
                else:
                    fitted_model = pipeline.fit(train[features], train["Log_Volume"])
                    params, cv_wape = {}, np.nan
                predicted = np.maximum(np.expm1(fitted_model.predict(test[features])), 0)
                metrics = regression_metrics(test["Volume"], predicted)
                key = f"{set_name}__{model_name}"
                fitted[key] = fitted_model
                rows.append({"Feature_Set": set_name, "Model": model_name, "N_Train": len(train),
                             "N_Test": len(test), "CV_WAPE": cv_wape, "WAPE": metrics["wape"],
                             "MAE": metrics["mae"], "RMSE": metrics["rmse"], "BIAS": metrics["bias"],
                             "ABS_BIAS": abs(metrics["bias"]), "Best_Params": params})
                prediction_frames.append(pd.DataFrame({"date": test["date"].values,
                    "Actual_Volume": test["Volume"].values, "Predicted_Volume": predicted,
                    "Feature_Set": set_name, "Model": model_name}))

        baseline_map = {"Naive_Lag_1": "Volume_lag_1", "Naive_Lag_12": "Volume_lag_12",
                        "Moving_Average_3": "Volume_MA_3"}
        baseline = data[data["date"].isin(test_dates)]
        for name, column in baseline_map.items():
            part = baseline[["date", "Volume", column]].dropna()
            metrics = regression_metrics(part["Volume"], part[column])
            rows.append({"Feature_Set": "Baseline", "Model": name, "N_Train": np.nan,
                         "N_Test": len(part), "CV_WAPE": np.nan, "WAPE": metrics["wape"],
                         "MAE": metrics["mae"], "RMSE": metrics["rmse"], "BIAS": metrics["bias"],
                         "ABS_BIAS": abs(metrics["bias"]), "Best_Params": {}})
            prediction_frames.append(pd.DataFrame({"date": part["date"], "Actual_Volume": part["Volume"],
                "Predicted_Volume": part[column], "Feature_Set": "Baseline", "Model": name}))

        results = pd.DataFrame(rows).sort_values(["WAPE", "ABS_BIAS"]).reset_index(drop=True)
        eligible = results[(results["Feature_Set"] != "Baseline") & (results["ABS_BIAS"] <= self.max_abs_bias)]
        candidates = eligible if len(eligible) else results[results["Feature_Set"] != "Baseline"]
        best = candidates.sort_values(["WAPE", "ABS_BIAS"]).iloc[0]
        key = f"{best['Feature_Set']}__{best['Model']}"
        predictions = pd.concat(prediction_frames, ignore_index=True)
        predictions["Error"] = predictions["Predicted_Volume"] - predictions["Actual_Volume"]
        predictions["Error_pct"] = predictions["Error"] / predictions["Actual_Volume"]
        return ExperimentOutput(results, predictions, fitted, best, fitted[key])
