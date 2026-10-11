import argparse
from pathlib import Path

from volume_forecasting.artifacts import ArtifactStore
from volume_forecasting.config import ConfigLoader
from volume_forecasting.data_loader import DataLoader
from volume_forecasting.feature_engineering import FeatureEngineer
from volume_forecasting.prediction import VolumePredictor
from volume_forecasting.scoring import OneStepScoringBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast Marca 2 volume one month ahead.")
    parser.add_argument("--forecast-date", required=True)
    parser.add_argument("--output", default="outputs/next_month_prediction.csv")
    args = parser.parse_args()

    config = ConfigLoader.load("config/model_config.json")
    sales, macro = DataLoader().load_sources(
        config["sales_path"], config["macro_path"], config.get("sales_sheet", "Marca")
    )
    engineered = FeatureEngineer(brand=config["brand"]).build(sales, macro)
    scoring_row = OneStepScoringBuilder().build(engineered, args.forecast_date)
    model, metadata = ArtifactStore.load(
        config["artifacts"]["model_path"], config["artifacts"]["metadata_path"]
    )
    prediction = VolumePredictor(model, metadata["features"]).predict(scoring_row)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(prediction.to_string(index=False))


if __name__ == "__main__":
    main()
