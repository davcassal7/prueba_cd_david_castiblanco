from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import joblib


class ArtifactStore:
    """Persists and loads the complete sklearn pipeline plus metadata."""

    @staticmethod
    def save(model: Any, metadata: dict[str, Any], model_path: str | Path, metadata_path: str | Path) -> None:
        model_file = Path(model_path)
        metadata_file = Path(metadata_path)
        model_file.parent.mkdir(parents=True, exist_ok=True)
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_file)
        payload = dict(metadata)
        payload["saved_at_utc"] = datetime.now(timezone.utc).isoformat()
        with metadata_file.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False, default=str)

    @staticmethod
    def load(model_path: str | Path, metadata_path: str | Path):
        model = joblib.load(model_path)
        with Path(metadata_path).open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        return model, metadata
