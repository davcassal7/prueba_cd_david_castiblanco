from pathlib import Path

from volume_forecasting.artifacts import ArtifactStore
from volume_forecasting.config import ConfigLoader
from volume_forecasting.data_loader import DataLoader
from volume_forecasting.feature_engineering import FeatureEngineer
from volume_forecasting.modeling import ModelTrainer


def main() -> None:
    config = ConfigLoader.load("config/model_config.json")
    sales, macro = DataLoader().load_sources(config["sales_path"], config["macro_path"])
    data = FeatureEngineer().build(sales, macro)

    trainer = ModelTrainer(
        features=config["features"],
        alpha=float(config["model"]["params"]["alpha"]),
        test_months=int(config["test_months"]),
    )
    result = trainer.fit_evaluate(data)
    result.predictions.to_csv("outputs/holdout_predictions.csv", index=False)

    final_model = trainer.refit_all(data)
    metadata = result.metadata | {"refit_on_all_available_data": True}
    ArtifactStore.save(
        final_model,
        metadata,
        config["artifacts"]["model_path"],
        config["artifacts"]["metadata_path"],
    )
    print("Model saved successfully")
    print(result.metrics)


if __name__ == "__main__":
    main()
