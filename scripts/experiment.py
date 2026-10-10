from volume_forecasting.artifacts import ArtifactStore
from volume_forecasting.config import ConfigLoader
from volume_forecasting.data_loader import DataLoader
from volume_forecasting.feature_engineering import FeatureEngineer
from volume_forecasting.experiment import ForecastExperiment


def main():
    config = ConfigLoader.load("config/model_config.json")
    sales, macro = DataLoader().load_sources(config["sales_path"], config["macro_path"])
    data = FeatureEngineer().build(sales, macro)
    feature_sets = {
        "compact": ["Volume_lag_1", "DN_lag_1", "Avg_Price_lag_1", "Is_December", "Low_Season"],
        "initial": ["Volume_lag_1", "DN_lag_1", "Avg_Price_lag_1", "IPC_lag_1", "Is_December", "Low_Season"],
        "moving_average": ["Volume_MA_3", "DN_lag_1", "Avg_Price_lag_1", "IPC_lag_1", "Is_December", "Low_Season"],
    }
    output = ForecastExperiment(feature_sets, test_months=12, max_abs_bias=0.07).run(data)
    output.results.to_csv("outputs/experiment_results.csv", index=False)
    output.predictions.to_csv("outputs/all_predictions.csv", index=False)
    best = output.best_result
    key = f"{best['Feature_Set']}__{best['Model']}"
    metadata = {"model_key": key, "feature_set": best["Feature_Set"],
                "features": feature_sets[best["Feature_Set"]], "target": "Log_Volume",
                "best_params": best["Best_Params"], "holdout_metrics": {
                    "wape": best["WAPE"], "bias": best["BIAS"], "mae": best["MAE"], "rmse": best["RMSE"]}}
    ArtifactStore.save(output.best_model, metadata, config["artifacts"]["model_path"],
                       config["artifacts"]["metadata_path"])
    print(best)


if __name__ == "__main__":
    main()
