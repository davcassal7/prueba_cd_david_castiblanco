import numpy as np
from volume_forecasting.metrics import wape, bias


def test_wape_zero_for_perfect_prediction():
    actual = np.array([100.0, 200.0])
    assert wape(actual, actual) == 0.0


def test_bias_sign():
    assert bias([100.0], [110.0]) > 0
    assert bias([100.0], [90.0]) < 0
