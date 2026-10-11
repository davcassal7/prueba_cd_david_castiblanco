from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Select the target brand and create leakage-safe monthly features."""

    REQUIRED_SALES = {"Año", "Mes", "Marca", "Volume", "NR", "DN"}
    REQUIRED_MACRO = {"date", "concept", "frequency", "value"}

    def __init__(self, brand: str = "Marca 2", low_season_months: tuple[int, ...] = (7, 10, 11)) -> None:
        self.brand = brand
        self.low_season_months = low_season_months

    @staticmethod
    def _validate_columns(data: pd.DataFrame, required: set[str], source: str) -> None:
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"{source} is missing columns: {sorted(missing)}")

    def select_brand(self, sales: pd.DataFrame) -> pd.DataFrame:
        self._validate_columns(sales, self.REQUIRED_SALES, "sales")
        selected = sales.loc[sales["Marca"].eq(self.brand)].copy()
        if selected.empty:
            raise ValueError(f"No observations found for {self.brand}.")
        if not selected["Marca"].eq(self.brand).all():
            raise ValueError("Brand filtering failed.")
        return selected

    @staticmethod
    def _validate_monthly_dates(data: pd.DataFrame) -> None:
        if data["date"].duplicated().any():
            raise ValueError("Duplicate monthly dates were found.")
        expected = pd.date_range(data["date"].min(), data["date"].max(), freq="MS")
        missing = expected.difference(data["date"])
        if len(missing):
            raise ValueError(f"Missing monthly dates: {missing.tolist()}")

    def transform_sales(self, sales: pd.DataFrame) -> pd.DataFrame:
        data = self.select_brand(sales)
        data["date"] = pd.to_datetime(dict(year=data["Año"], month=data["Mes"], day=1))
        data = data.sort_values("date").reset_index(drop=True)
        self._validate_monthly_dates(data)

        data["Volume_lag_1"] = data["Volume"].shift(1)
        data["Volume_lag_12"] = data["Volume"].shift(12)
        data["Volume_MA_3"] = data["Volume"].shift(1).rolling(3, min_periods=3).mean()
        data["DN_lag_1"] = data["DN"].shift(1)
        data["Avg_Price"] = data["NR"].div(data["Volume"].replace(0, np.nan))
        data["Avg_Price_lag_1"] = data["Avg_Price"].shift(1)
        data["Trend"] = np.arange(len(data), dtype=float)
        data["Is_February"] = data["date"].dt.month.eq(2).astype(int)
        data["Is_December"] = data["date"].dt.month.eq(12).astype(int)
        data["Low_Season"] = data["date"].dt.month.isin(self.low_season_months).astype(int)
        data["Log_Volume"] = np.log1p(data["Volume"])
        return data

    def transform_cpi(self, macro: pd.DataFrame) -> pd.DataFrame:
        self._validate_columns(macro, self.REQUIRED_MACRO, "macro")
        cpi = macro.loc[
            macro["concept"].eq("CPI") & macro["frequency"].eq("monthly"),
            ["date", "value"],
        ].copy()
        cpi["date"] = pd.to_datetime(cpi["date"]).dt.to_period("M").dt.to_timestamp()
        cpi = cpi.rename(columns={"value": "IPC"}).sort_values("date").reset_index(drop=True)
        if cpi["date"].duplicated().any():
            raise ValueError("Multiple monthly CPI observations were found for one date.")
        cpi["IPC_lag_1"] = cpi["IPC"].shift(1)
        return cpi[["date", "IPC_lag_1"]]

    def build(self, sales: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
        sales_features = self.transform_sales(sales)
        cpi_features = self.transform_cpi(macro)
        result = sales_features.merge(cpi_features, on="date", how="left", validate="one_to_one")
        if not result["Marca"].eq(self.brand).all():
            raise ValueError("The final modeling table contains another brand.")
        return result
