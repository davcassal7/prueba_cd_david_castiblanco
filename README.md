# Volume Forecasting - Marca 2

Pronóstico mensual de volumen para la marca principal del archivo entregado. `Marca 2` fue seleccionada por tener el mayor volumen acumulado, aproximadamente 39.8% del total del periodo.

## Hallazgos analíticos

- Periodo: enero de 2023 a diciembre de 2025.
- Fuerza estacional: 0.464, clasificada como moderada.
- Diciembre presenta señal positiva; julio, octubre y noviembre se tratan como temporada baja.
- La serie presenta una tendencia descendente pronunciada.
- `Volume_lag_12` no funciona bien como baseline por la caída estructural.

## Modelos y selección

Se comparan regresión lineal, Ridge, Elastic Net, Random Forest y Gradient Boosting con validación temporal y holdout común de 12 meses.

**Mejor precisión:** `seasonal_february + LinearRegression`, WAPE 15.44%, Bias +9.22%.

**Modelo operativo seleccionado:** `moving_average_trend + ElasticNet`, WAPE 17.87%, Bias -2.69%, MAE 6,683 hl y RMSE 7,685 hl. Se elige por tener el menor WAPE entre modelos con Bias absoluto <= 7%.

Variables productivas:

- `Volume_MA_3`
- `DN_lag_1`
- `Avg_Price_lag_1`
- `Trend`
- `Is_December`
- `Low_Season`

## Estructura

- `notebooks/02_experimentation.ipynb`: orquesta y documenta la experimentación.
- `src/volume_forecasting/`: implementación modular OOP.
- `scripts/experiment.py`: experimento completo, selección, refit y exportación.
- `scripts/forecast_next_month.py`: forecast de un paso.
- `models/`: pipeline final y metadata.
- `outputs/`: ranking, predicciones y coeficientes.
- `presentacion/`: presentacion ejecutiva.
- `tests/`: pruebas unitarias.

## Ejecución

```bash
python -m pip install -e .
python scripts/experiment.py
```

Para pronosticar enero de 2026 después de entrenar:

```bash
python scripts/forecast_next_month.py --forecast-date 2026-01-01
```

Solo se admite un paso futuro porque `DN` y precio promedio entran rezagados. Los coeficientes se interpretan como asociaciones predictivas, no efectos causales de precio.
