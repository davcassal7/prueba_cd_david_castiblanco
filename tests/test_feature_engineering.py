import pandas as pd
from volume_forecasting.feature_engineering import FeatureEngineer


def test_sales_features_are_lagged():
    sales = pd.DataFrame({
        "Año": [2023, 2023, 2023, 2023],
        "Mes": [1, 2, 3, 4],
        "Volume": [100.0, 110.0, 120.0, 130.0],
        "NR": [1000.0, 1210.0, 1440.0, 1690.0],
        "DN": [10.0, 11.0, 12.0, 13.0],
    })
    result = FeatureEngineer().transform_sales(sales)
    assert result.loc[1, "Volume_lag_1"] == 100.0
    assert result.loc[3, "Volume_MA_3"] == 110.0
