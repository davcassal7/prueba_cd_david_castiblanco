import pandas as pd
from volume_forecasting.scoring import OneStepScoringBuilder


def test_next_month_features():
    history = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=3, freq="MS"),
                            "Volume": [100., 110., 120.], "DN": [10., 11., 12.],
                            "Avg_Price": [5., 6., 7.], "IPC": [101., 102., 103.]})
    row = OneStepScoringBuilder().build(history, "2025-04-01")
    assert row.loc[0, "Volume_MA_3"] == 110.
    assert row.loc[0, "DN_lag_1"] == 12.
