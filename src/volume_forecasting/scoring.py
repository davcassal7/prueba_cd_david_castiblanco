from __future__ import annotations

import pandas as pd


class OneStepScoringBuilder:
    """Build one future feature row from observed Marca 2 history."""

    def build(self, engineered_history: pd.DataFrame, forecast_date) -> pd.DataFrame:
        history = engineered_history.sort_values("date").copy()
        forecast_date = pd.Timestamp(forecast_date).to_period("M").to_timestamp()
        expected = history["date"].max() + pd.offsets.MonthBegin(1)
        if forecast_date != expected:
            raise ValueError(f"Only one-step forecasts are supported. Expected {expected.date()}.")
        if len(history) < 3:
            raise ValueError("At least three historical months are required.")
        if "Marca" in history and not history["Marca"].eq("Marca 2").all():
            raise ValueError("Scoring history must contain only Marca 2.")
        latest = history.iloc[-1]
        return pd.DataFrame([{
            "date": forecast_date,
            "Volume_MA_3": history["Volume"].tail(3).mean(),
            "DN_lag_1": latest["DN"],
            "Avg_Price_lag_1": latest["Avg_Price"],
            "Trend": float(len(history)),
            "Is_December": int(forecast_date.month == 12),
            "Low_Season": int(forecast_date.month in (7, 10, 11)),
        }])
