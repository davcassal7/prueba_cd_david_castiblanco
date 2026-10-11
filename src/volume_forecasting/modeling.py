from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone


class ProductionModelTrainer:
    """Refit a selected sklearn pipeline on all available modeling rows."""

    @staticmethod
    def refit(model, data: pd.DataFrame, features: list[str], target: str = "Log_Volume"):
        columns = [target, *features]
        clean = data[columns].replace([np.inf, -np.inf], np.nan).dropna()
        production_model = clone(model)
        production_model.fit(clean[features], clean[target])
        return production_model, len(clean)
