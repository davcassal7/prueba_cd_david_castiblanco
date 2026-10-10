# Cell to add at the end of the experimentation notebook.
# It exports the already fitted winner and a JSON contract.
from volume_forecasting.artifacts import ArtifactStore

MODEL_KEY = "moving_average__Ridge"
best_model = fitted_models[MODEL_KEY]
metadata = {
    "model_name": "Ridge",
    "feature_set": "moving_average",
    "features": feature_sets["moving_average"],
    "target": "Log_Volume",
    "best_params": {"model__alpha": 10.0},
    "holdout_metrics": {
        "wape": 0.12859263,
        "bias": 0.06560567,
        "mae": 17219.186671,
        "rmse": 21097.135172,
    },
}
ArtifactStore.save(
    best_model,
    metadata,
    "models/best_model.joblib",
    "models/model_metadata.json",
)
