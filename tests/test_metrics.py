from volume_forecasting.metrics import wape, bias


def test_perfect_prediction_has_zero_wape():
    assert wape([100.0, 200.0], [100.0, 200.0]) == 0.0


def test_bias_sign():
    assert bias([100.0], [110.0]) > 0
    assert bias([100.0], [90.0]) < 0
