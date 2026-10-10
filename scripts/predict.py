import argparse
from pathlib import Path
import pandas as pd

from volume_forecasting.artifacts import ArtifactStore
from volume_forecasting.config import ConfigLoader
from volume_forecasting.prediction import VolumePredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate monthly volume predictions")
    parser.add_argument("--input", required=True, help="CSV/XLSX containing engineered features")
    parser.add_argument("--output", default=None, help="Prediction CSV path")
    args = parser.parse_args()

    config = ConfigLoader.load("config/model_config.json")
    model, metadata = ArtifactStore.load(
        config["artifacts"]["model_path"],
        config["artifacts"]["metadata_path"],
    )
    path = Path(args.input)
    data = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"])
    predictions = VolumePredictor(model, metadata["features"]).predict(data)
    output = args.output or config["prediction_output"]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)
    print(f"Predictions saved to {output}")


if __name__ == "__main__":
    main()
