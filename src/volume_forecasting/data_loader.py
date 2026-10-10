from __future__ import annotations

from pathlib import Path
import pandas as pd


class DataLoader:
    """Loads sales and macroeconomic source files."""

    @staticmethod
    def read(path: str | Path) -> pd.DataFrame:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        suffix = file_path.suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(file_path)
        if suffix == ".csv":
            return pd.read_csv(file_path)
        raise ValueError(f"Unsupported file extension: {suffix}")

    def load_sources(self, sales_path: str | Path, macro_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.read(sales_path), self.read(macro_path)
