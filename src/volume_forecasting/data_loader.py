from __future__ import annotations

from pathlib import Path
import pandas as pd


class DataLoader:
    """Load commercial and macroeconomic source files."""

    @staticmethod
    def read(path: str | Path, sheet_name: str | int = 0) -> pd.DataFrame:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        if file_path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(file_path, sheet_name=sheet_name)
        if file_path.suffix.lower() == ".csv":
            return pd.read_csv(file_path)
        raise ValueError(f"Unsupported input extension: {file_path.suffix}")

    def load_sources(
        self,
        sales_path: str | Path,
        macro_path: str | Path,
        sales_sheet: str = "Marca",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.read(sales_path, sales_sheet), self.read(macro_path)
