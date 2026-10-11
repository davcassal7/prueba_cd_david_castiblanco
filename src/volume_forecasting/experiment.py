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
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer

from .metrics import wape, regression_metrics


@dataclass
class ExperimentOutput:
    """Container for experiment rankings, predictions and fitted estimators."""

    results: pd.DataFrame
    predictions: pd.DataFrame
    fitted_models: dict[str, Any]
    selected_result: pd.Series
    accuracy_result: pd.Series
    selected_model: Any


class ForecastExperiment:
    """Compare feature sets and regressors using a shared chronological holdout."""

    def __init__(
        self,
        feature_sets: dict[str, list[str]],
        test_months: int = 12,
        max_abs_bias: float = 0.07,
        random_state: int = 42,
    ) -> None:
        self.feature_sets = feature_sets
        self.test_months = test_months
        self.max_abs_bias = max_abs_bias
        self.random_state = random_state

    @staticmethod
    def _linear_pipeline(model) -> Pipeline:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ])

    def model_configs(self) -> dict[str, dict[str, Any]]:
        return {
            "LinearRegression": {
                "pipeline": self._linear_pipeline(LinearRegression()),
                "param_grid": {},
            },
            "Ridge": {
                "pipeline": self._linear_pipeline(Ridge()),
                "param_grid": {"model__alpha": [0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0]},
            },
            "ElasticNet": {
                "pipeline": self._linear_pipeline(
                    ElasticNet(max_iter=20000, random_state=self.random_state)
                ),
                "param_grid": {
                    "model__alpha": [0.01, 0.05, 0.1, 0.5, 1.0],
                    "model__l1_ratio": [0.1, 0.25, 0.5, 0.75],
                },
            },
            "RandomForest": {
                "pipeline": Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", RandomForestRegressor(random_state=self.random_state, n_jobs=-1)),
                ]),
                "param_grid": {
                    "model__n_estimators": [100, 300],
                    "model__max_depth": [2, 3],
                    "model__min_samples_leaf": [3, 5],
                    "model__max_features": [0.7, 1.0],
                },
            },
            "GradientBoosting": {
                "pipeline": Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", GradientBoostingRegressor(random_state=self.random_state)),
                ]),
                "param_grid": {
                    "model__n_estimators": [20, 50, 100],
                    "model__learning_rate": [0.01, 0.03, 0.05],
                    "model__max_depth": [1, 2],
                    "model__min_samples_leaf": [3, 5],
                    "model__loss": ["squared_error", "huber"],
                },
            },
        }

    @staticmethod
    def feature_sets_default() -> dict[str, list[str]]:
        return {
            "compact": [
                "Volume_lag_1", "DN_lag_1", "Avg_Price_lag_1",
                "Is_December", "Low_Season",
            ],
            "initial": [
                "Volume_lag_1", "DN_lag_1", "Avg_Price_lag_1", "IPC_lag_1",
                "Is_December", "Low_Season",
            ],
            "moving_average": [
                "Volume_MA_3", "DN_lag_1", "Avg_Price_lag_1", "IPC_lag_1",
                "Is_December", "Low_Season",
            ],
            "seasonal_february": [
                "Volume_MA_3", "DN_lag_1", "Avg_Price_lag_1", "IPC_lag_1",
                "Is_February", "Is_December", "Low_Season",
            ],
            "moving_average_trend": [
                "Volume_MA_3", "DN_lag_1", "Avg_Price_lag_1", "Trend",
                "Is_December", "Low_Season",
            ],
        }

    @staticmethod
    def _log_wape(y_true_log, y_pred_log) -> float:
        return wape(np.expm1(y_true_log), np.maximum(np.expm1(y_pred_log), 0))

    @staticmethod
    def _time_cv(n_train: int) -> TimeSeriesSplit:
        splits = 5 if n_train >= 24 else 3 if n_train >= 16 else 2
        return TimeSeriesSplit(n_splits=splits, test_size=1)

    def _test_dates(self, data: pd.DataFrame) -> pd.DatetimeIndex:
        dates = data["date"].dropna().sort_values().unique()
        if len(dates) <= self.test_months:
            raise ValueError("Insufficient observations for the requested holdout.")
        return pd.DatetimeIndex(dates[-self.test_months:])

    @staticmethod
    def _split(data: pd.DataFrame, features: list[str], test_dates: pd.DatetimeIndex):
        columns = ["date", "Volume", "Log_Volume", *features]
        clean = data[columns].replace([np.inf, -np.inf], np.nan).dropna().sort_values("date")
        train = clean[clean["date"] < test_dates.min()].copy()
        test = clean[clean["date"].isin(test_dates)].copy()
        if len(test) != len(test_dates):
            raise ValueError("The feature set is incomplete for the shared holdout.")
        if len(train) < 10:
            raise ValueError("The feature set has fewer than ten training observations.")
        return train, test

    def run(self, data: pd.DataFrame) -> ExperimentOutput:
        if "Marca" in data and not data["Marca"].eq("Marca 2").all():
            raise ValueError("Experiment input must contain only Marca 2.")

        scorer = make_scorer(self._log_wape, greater_is_better=False)
        test_dates = self._test_dates(data)
        results, prediction_frames, fitted = [], [], {}

        for set_name, features in self.feature_sets.items():
            train, test = self._split(data, features, test_dates)
            cv = self._time_cv(len(train))
            for model_name, config in self.model_configs().items():
                pipeline = clone(config["pipeline"])
                if config["param_grid"]:
                    search = GridSearchCV(
                        pipeline,
                        config["param_grid"],
                        scoring=scorer,
                        cv=cv,
                        n_jobs=-1,
                        refit=True,
                        error_score="raise",
                    )
                    search.fit(train[features], train["Log_Volume"])
                    fitted_model = search.best_estimator_
                    params = search.best_params_
                    cv_wape = -search.best_score_
                else:
                    scores = cross_val_score(
                        pipeline,
                        train[features],
                        train["Log_Volume"],
                        scoring=scorer,
                        cv=cv,
                        n_jobs=-1,
                    )
                    cv_wape = -scores.mean()
                    params = {}
                    fitted_model = pipeline.fit(train[features], train["Log_Volume"])

                predicted = np.maximum(np.expm1(fitted_model.predict(test[features])), 0)
                metrics = regression_metrics(test["Volume"], predicted)
                key = f"{set_name}__{model_name}"
                fitted[key] = fitted_model
                results.append({
                    "Feature_Set": set_name,
                    "Model": model_name,
                    "N_Train": len(train),
                    "N_Test": len(test),
                    "CV_WAPE": cv_wape,
                    "WAPE": metrics["wape"],
                    "MAE": metrics["mae"],
                    "RMSE": metrics["rmse"],
                    "BIAS": metrics["bias"],
                    "ABS_BIAS": abs(metrics["bias"]),
                    "Best_Params": params,
                })
                prediction_frames.append(pd.DataFrame({
                    "date": test["date"].values,
                    "Actual_Volume": test["Volume"].values,
                    "Predicted_Volume": predicted,
                    "Feature_Set": set_name,
                    "Model": model_name,
                }))

        baseline_columns = {
            "Naive_Lag_1": "Volume_lag_1",
            "Naive_Lag_12": "Volume_lag_12",
            "Moving_Average_3": "Volume_MA_3",
        }
        holdout = data[data["date"].isin(test_dates)].sort_values("date")
        for name, column in baseline_columns.items():
            part = holdout[["date", "Volume", column]].dropna()
            metrics = regression_metrics(part["Volume"], part[column])
            results.append({
                "Feature_Set": "Baseline", "Model": name, "N_Train": np.nan,
                "N_Test": len(part), "CV_WAPE": np.nan, "WAPE": metrics["wape"],
                "MAE": metrics["mae"], "RMSE": metrics["rmse"], "BIAS": metrics["bias"],
                "ABS_BIAS": abs(metrics["bias"]), "Best_Params": {},
            })
            prediction_frames.append(pd.DataFrame({
                "date": part["date"].values,
                "Actual_Volume": part["Volume"].values,
                "Predicted_Volume": part[column].values,
                "Feature_Set": "Baseline",
                "Model": name,
            }))

        result_table = pd.DataFrame(results).sort_values(["WAPE", "ABS_BIAS"]).reset_index(drop=True)
        model_rows = result_table[result_table["Feature_Set"] != "Baseline"].copy()
        accuracy_result = model_rows.sort_values(["WAPE", "ABS_BIAS"]).iloc[0]
        eligible = model_rows[model_rows["ABS_BIAS"] <= self.max_abs_bias]
        selection_pool = eligible if len(eligible) else model_rows
        selected_result = selection_pool.sort_values(["WAPE", "ABS_BIAS", "RMSE"]).iloc[0]
        selected_key = f"{selected_result['Feature_Set']}__{selected_result['Model']}"

        predictions = pd.concat(prediction_frames, ignore_index=True)
        predictions["Error"] = predictions["Predicted_Volume"] - predictions["Actual_Volume"]
        predictions["Error_pct"] = predictions["Error"] / predictions["Actual_Volume"]
        predictions["Absolute_Error_pct"] = predictions["Error_pct"].abs()

        return ExperimentOutput(
            results=result_table,
            predictions=predictions,
            fitted_models=fitted,
            selected_result=selected_result,
            accuracy_result=accuracy_result,
            selected_model=fitted[selected_key],
        )
