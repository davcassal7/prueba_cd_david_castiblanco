from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd


class ArtifactStore:
    """Persist and load a fitted pipeline and its readable metadata."""

    @staticmethod
    def _json_default(value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        raise TypeError(f"Unsupported JSON type: {type(value)}")

    @classmethod
    def save(cls, model: Any, metadata: dict[str, Any], model_path: str | Path, metadata_path: str | Path) -> None:
        model_file = Path(model_path)
        metadata_file = Path(metadata_path)
        model_file.parent.mkdir(parents=True, exist_ok=True)
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_file)
        payload = dict(metadata)
        payload["saved_at_utc"] = datetime.now(timezone.utc).isoformat()
        with metadata_file.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False, default=cls._json_default)

    @staticmethod
    def load(model_path: str | Path, metadata_path: str | Path):
        model = joblib.load(model_path)
        with Path(metadata_path).open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        return model, metadata
