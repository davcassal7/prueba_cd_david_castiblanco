from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ConfigLoader:
    """Loads and validates a JSON project configuration."""

    REQUIRED_KEYS = {"sales_path", "macro_path", "features", "model", "artifacts"}

    @classmethod
    def load(cls, path: str | Path) -> dict[str, Any]:
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)
        missing = cls.REQUIRED_KEYS.difference(config)
        if missing:
            raise ValueError(f"Missing configuration keys: {sorted(missing)}")
        return config
