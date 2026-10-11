from __future__ import annotations

import numpy as np
import pandas as pd


class VolumePredictor:
    """Generate volume predictions using a fitted log-target pipeline."""

    def __init__(self, model, features: list[str]) -> None:
        self.model = model
        self.features = features

    def predict(self, feature_data: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.features).difference(feature_data.columns)
        if missing:
            raise ValueError(f"Prediction data is missing features: {sorted(missing)}")
        prediction = np.maximum(np.expm1(self.model.predict(feature_data[self.features])), 0)
        output = pd.DataFrame({"predicted_volume": prediction}, index=feature_data.index)
        if "date" in feature_data.columns:
            output.insert(0, "date", feature_data["date"].values)
        return output
