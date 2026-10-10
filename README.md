# Volume Forecasting

Proyecto OOP para pronosticar volumen mensual de ventas usando variables comerciales, estacionales y macroeconómicas.

## Resultado seleccionado

- Modelo: Ridge
- Feature set: `moving_average`
- Alpha: `10.0`
- WAPE holdout: `12.86%`
- Bias holdout: `6.56%`
- Mejor baseline: media móvil de 3 meses, WAPE `16.15%`

## Estructura

- `notebooks/`: EDA y experimentación. No contienen lógica necesaria para producción.
- `src/volume_forecasting/`: paquete modular OOP.
- `scripts/train.py`: entrena, evalúa, reentrena con toda la historia y persiste artefactos.
- `scripts/predict.py`: carga el modelo y genera un CSV de predicciones.
- `config/model_config.json`: rutas, variables y parámetros.
- `models/`: artefactos generados con joblib y JSON.
- `outputs/`: métricas y predicciones.
- `tests/`: pruebas unitarias mínimas.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Ubique los dos Excel en `data/raw/` con los nombres definidos en `config/model_config.json`.

## Entrenamiento y exportación

```bash
python scripts/train.py
```

Genera:

- `models/best_model.joblib`: pipeline completo, incluyendo imputación, escalamiento y Ridge.
- `models/model_metadata.json`: variables, parámetros, fechas y métricas.
- `outputs/holdout_predictions.csv`: predicciones del holdout.

## Predicción

El archivo de entrada debe contener las variables ya construidas definidas en el metadata del modelo.

```bash
python scripts/predict.py --input data/scoring_features.xlsx --output outputs/predictions.csv
```

## Pruebas

```bash
pytest -q
```

## Nota operativa

Para pronosticar un mes futuro se requieren los valores conocidos al inicio del mes: media móvil del volumen, DN rezagado, precio promedio rezagado e IPC rezagado. El script de scoring no debe recibir valores contemporáneos desconocidos.

## Experimentación completa

```bash
python scripts/experiment.py
```

Compara los tres feature sets, cinco algoritmos y tres baselines del notebook original. Guarda resultados, predicciones, el pipeline ganador y metadata.

## Forecast end-to-end del siguiente mes

```bash
python scripts/forecast_next_month.py --forecast-date 2026-01-01 --output outputs/next_month_prediction.csv
```

El comando carga fuentes históricas, crea features, carga el modelo persistido y escribe la predicción. Solo admite un paso al futuro porque DN, precio e IPC se usan rezagados.
