# 📊 Análisis y Pronóstico de Cortes de Luz mediante Series Temporales

Este repositorio contiene el desarrollo completo de un análisis de series temporales aplicado al estudio y pronóstico de cortes de luz, utilizando como variable proxy el índice de búsquedas en Google Trends. El trabajo combina enfoques estadísticos clásicos y modelos de machine learning y deep learning, con el objetivo de analizar la dinámica temporal del fenómeno, evaluar el aporte de variables meteorológicas y generar pronósticos de corto plazo.

---

## 🧠 Objetivo del proyecto

Modelar y pronosticar la evolución de los cortes de luz explorando:

- La dinámica propia del indicador de cortes.
- El aporte predictivo de variables meteorológicas (temperatura y humedad).
- La interacción dinámica entre las series mediante modelos multivariados.
- La capacidad de distintos enfoques para detectar eventos extremos.

---

## 📁 Estructura del repositorio

```
├── scripts_modelos/         # Scripts individuales por modelo
├── resultados_modelos/      # CSVs con predicciones y métricas por modelo
├── completo.csv             # Dataset (262 observaciones semanales)
├── TP_series_grupo3.ipynb   # Notebook Parte 1
├── TP2_series_grupo_3.ipynb # Notebook Parte 2
└── README.md
```

---

## 📈 Metodología

### Parte 1 — Modelos Estadísticos Clásicos

**1. Análisis exploratorio**
Visualización de series, identificación de estacionalidad y tendencias, análisis descriptivo.

**2. Estacionariedad**
Tests ADF y KPSS para verificar condiciones de estacionariedad.

**3. Modelos univariados**
SARIMA como línea base y AutoARIMA para selección automática de parámetros. Evaluación mediante MAE, RMSE y MAPE.

**4. Modelos con variables exógenas**
SARIMAX con temperatura y humedad como covariables. Comparación con modelos univariados.

**5. Modelos multivariados**
VAR en niveles con selección de rezagos, tests de causalidad de Granger y funciones Impulso-Respuesta (IRF).

**6. Pronósticos**
Pronósticos finales con modelos univariados y conjuntos con VAR. Intervalos de confianza y validación con esquemas rolling window.

---

### Parte 2 — Machine Learning y Deep Learning

Split de evaluación unificado: 80% entrenamiento / 20% test para todos los modelos.

**1. SVR (Support Vector Regression)**
Modelos univariados para Cortes, Temp y Hum, y modelo multivariado Temp+Hum → Cortes. Optimización de hiperparámetros mediante GridSearchCV con TimeSeriesSplit. Predicción recursiva out-of-sample.

**2. Darts**
Suavizado Exponencial, FFT y Transformer. Comparación de enfoques clásicos y basados en atención dentro del mismo framework.

**3. Prophet**
Modelo aditivo con estacionalidad anual y regressors externos (Temp y Hum). Optimización de hiperparámetros mediante grid search manual sobre las tres series.

**4. LSTM**
Red neuronal recurrente con arquitectura LSTM(40) + Dropout(0.20) + Dense(1). Modelos univariados para las tres series y modelo multivariado Temp+Hum → Cortes. Predicción recursiva con ventana de 4 semanas.

**5. H2O AutoML**
Búsqueda automática sobre XGBoost, Random Forest, GLM y Stacked Ensembles. Grid search externo sobre número de lags y tiempo de búsqueda. Modelos univariados y multivariado.

**6. Comparación global**
Evaluación conjunta de todos los modelos mediante RMSE, MAE, Var Ratio y Peak Recall. Análisis operativo orientado a la detección de eventos extremos.

---

## 👥 Grupo 3

**Materia:** Análisis de Series de Tiempo  
**Alumnos:** Arenas - Banegas - Marín - Zamora

---

## 🛠 Tecnologías utilizadas

Python 3.11 · scikit-learn · darts · neuralprophet · tensorflow · h2o · prophet · pandas · matplotlib
