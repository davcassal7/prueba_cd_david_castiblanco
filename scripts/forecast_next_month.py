import argparse
from pathlib import Path

from volume_forecasting.artifacts import ArtifactStore
from volume_forecasting.config import ConfigLoader
from volume_forecasting.data_loader import DataLoader
from volume_forecasting.feature_engineering import FeatureEngineer
from volume_forecasting.prediction import VolumePredictor
from volume_forecasting.scoring import OneStepScoringBuilder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast-date", required=True, help="First day or any date in next month")
    parser.add_argument("--output", default="outputs/next_month_prediction.csv")
    parser.add_argument("--ipc-lag-1", type=float, default=None,
                        help="Latest CPI known at forecast date; defaults to latest CPI in source")
    args = parser.parse_args()
    config = ConfigLoader.load("config/model_config.json")
    sales, macro = DataLoader().load_sources(config["sales_path"], config["macro_path"])
    engineered = FeatureEngineer().build(sales, macro)
    row = OneStepScoringBuilder().build(engineered, args.forecast_date, args.ipc_lag_1)
    model, metadata = ArtifactStore.load(config["artifacts"]["model_path"],
                                         config["artifacts"]["metadata_path"])
    prediction = VolumePredictor(model, metadata["features"]).predict(row)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(args.output, index=False)
    print(prediction.to_string(index=False))


if __name__ == "__main__":
    main()
