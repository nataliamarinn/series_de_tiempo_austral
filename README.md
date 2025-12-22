# 📊 Análisis y Pronóstico de Cortes de Luz mediante Series Temporales

Este repositorio contiene el desarrollo completo de un análisis de series temporales aplicado al estudio y pronóstico de cortes de luz, utilizando como variable proxy el índice de búsquedas en Google Trends. El trabajo combina enfoques univariados y multivariados con el objetivo de analizar la dinámica temporal del fenómeno, evaluar el aporte de variables meteorológicas y generar pronósticos de corto plazo.

# 🧠 Objetivo del proyecto

El objetivo principal es modelar y pronosticar la evolución de los cortes de luz, explorando:
- La dinámica propia del indicador de cortes.
- El aporte predictivo de variables meteorológicas (temperatura y humedad).
- La interacción dinámica entre las series mediante modelos multivariados.
- La estabilidad del desempeño predictivo bajo distintos esquemas de validación.

# 📈 Metodología

El análisis se desarrolla en las siguientes etapas:

## 1. Análisis exploratorio

- Visualización de series.
- Identificación de estacionalidad y tendencias.
- Análisis descriptivo.

## 2. Estacionariedad

-Test ADF y KPSS.


## 3. Modelos univariados

- SARIMA como línea base.
- AutoARIMA para selección automática de modelos.
- Evaluación mediante MAE, RMSE y MAPE.

## 4. Modelos con variables exógenas

- SARIMAX con temperatura y humedad.
- Comparación con modelos univariados.

## 5. Modelos multivariados

- VAR en niveles (series estacionarias).
- Selección de rezagos.
- Tests de causalidad de Granger.
- Funciones Impulso–Respuesta (IRF).

## 6. Pronósticos

- Pronósticos finales con modelos univariados.
-Pronósticos conjuntos con VAR.

Intervalos de confianza.

Validación con esquemas rolling window.
