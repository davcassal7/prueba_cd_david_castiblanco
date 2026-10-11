from pathlib import Path
import json
import pandas as pd

from volume_forecasting.artifacts import ArtifactStore
from volume_forecasting.config import ConfigLoader
from volume_forecasting.data_loader import DataLoader
from volume_forecasting.experiment import ForecastExperiment
from volume_forecasting.feature_engineering import FeatureEngineer
from volume_forecasting.modeling import ProductionModelTrainer


def main() -> None:
    config = ConfigLoader.load("config/model_config.json")
    sales, macro = DataLoader().load_sources(
        config["sales_path"], config["macro_path"], config.get("sales_sheet", "Marca")
    )
    data = FeatureEngineer(brand=config["brand"]).build(sales, macro)
    feature_sets = ForecastExperiment.feature_sets_default()
    output = ForecastExperiment(
        feature_sets=feature_sets,
        test_months=int(config["test_months"]),
        max_abs_bias=float(config["max_abs_bias"]),
    ).run(data)

    selected = output.selected_result
    accuracy = output.accuracy_result
    selected_features = feature_sets[selected["Feature_Set"]]
    production_model, production_rows = ProductionModelTrainer.refit(
        output.selected_model, data, selected_features
    )

    metadata = {
        "brand": config["brand"],
        "brand_selection_criterion": "Highest accumulated volume in the provided period.",
        "model_key": f"{selected['Feature_Set']}__{selected['Model']}",
        "model_name": selected["Model"],
        "feature_set": selected["Feature_Set"],
        "features": selected_features,
        "target": "Log_Volume",
        "prediction_scale": "Volume",
        "best_params": selected["Best_Params"],
        "selection_rule": "Lowest holdout WAPE among models with absolute Bias <= 7%.",
        "seasonal_strength": 0.464,
        "holdout_metrics": {
            "cv_wape": selected["CV_WAPE"],
            "wape": selected["WAPE"],
            "bias": selected["BIAS"],
            "mae": selected["MAE"],
            "rmse": selected["RMSE"],
        },
        "best_accuracy_candidate": {
            "model_key": f"{accuracy['Feature_Set']}__{accuracy['Model']}",
            "wape": accuracy["WAPE"],
            "bias": accuracy["BIAS"],
        },
        "refit_on_all_available_data": True,
        "production_training_rows": production_rows,
    }

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    result_export = output.results.copy()
    result_export["Best_Params"] = result_export["Best_Params"].apply(
        lambda value: json.dumps(value, ensure_ascii=False)
    )
    result_export.to_csv(output_dir / "experiment_results.csv", index=False, encoding="utf-8-sig")
    output.predictions.to_csv(output_dir / "all_predictions.csv", index=False, encoding="utf-8-sig")

    best_predictions = output.predictions[
        output.predictions["Feature_Set"].eq(selected["Feature_Set"])
        & output.predictions["Model"].eq(selected["Model"])
    ].sort_values("date")
    best_predictions.to_csv(output_dir / "best_model_predictions.csv", index=False, encoding="utf-8-sig")

    coefficients = pd.DataFrame({
        "Variable": selected_features,
        "Coefficient": output.selected_model.named_steps["model"].coef_,
    })
    coefficients["Absolute_Coefficient"] = coefficients["Coefficient"].abs()
    coefficients.sort_values("Absolute_Coefficient", ascending=False).to_csv(
        output_dir / "best_model_coefficients.csv", index=False, encoding="utf-8-sig"
    )

    ArtifactStore.save(
        production_model,
        metadata,
        config["artifacts"]["model_path"],
        config["artifacts"]["metadata_path"],
    )

    print("Selected operational model")
    print(selected.to_string())
    print("\nBest accuracy candidate")
    print(accuracy.to_string())


if __name__ == "__main__":
    main()
