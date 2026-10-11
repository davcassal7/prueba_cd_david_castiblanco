import pandas as pd
from volume_forecasting.scoring import OneStepScoringBuilder


def test_scoring_row_matches_selected_feature_contract():
    history = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=3, freq="MS"),
        "Marca": ["Marca 2"] * 3,
        "Volume": [100.0, 110.0, 120.0],
        "DN": [10.0, 11.0, 12.0],
        "Avg_Price": [5.0, 6.0, 7.0],
    })
    row = OneStepScoringBuilder().build(history, "2025-04-01")
    assert row.loc[0, "Volume_MA_3"] == 110.0
    assert row.loc[0, "DN_lag_1"] == 12.0
    assert row.loc[0, "Trend"] == 3.0
    assert "IPC_lag_1" not in row.columns
