import numpy as np
import pandas as pd
from volume_forecasting.feature_engineering import FeatureEngineer


def sample_sales():
    rows = []
    for brand, factor in [("Marca 1", 1.0), ("Marca 2", 2.0)]:
        for month in range(1, 5):
            rows.append({"Año": 2023, "Mes": month, "Marca": brand,
                         "Volume": 100.0 * month * factor,
                         "NR": 10000.0 * month * factor,
                         "DN": 10 * month})
    return pd.DataFrame(rows)


def test_selects_only_marca_2_and_builds_features():
    result = FeatureEngineer(brand="Marca 2").transform_sales(sample_sales())
    assert result["Marca"].eq("Marca 2").all()
    assert result.loc[1, "Volume_lag_1"] == 200.0
    assert result.loc[3, "Volume_MA_3"] == 400.0
    assert result.loc[3, "Trend"] == 3.0
    assert "Is_February" in result.columns
