# Bitácora de apoyo con IA — Rol 6: MLP

**Fecha:** 13 de noviembre de 2026  
**Proyecto:** Práctica 3 de Machine Learning  
**Responsabilidad:** MLP, ajuste de hiperparámetros y comparación de tiempos medidos con Big-O.

> Este documento es una bitácora resumida de la asistencia. No es una transcripción literal de todos los mensajes ni de las llamadas a herramientas.

## Solicitud y alcance

El estudiante pidió integrar su parte 6 en el repositorio del equipo. Se consultó el plan de trabajo y se identificaron estos entregables: fábrica `crear_mlp()`, búsqueda de parámetros, tabla comparativa, curva de pérdida, medición de escalabilidad y explicación de complejidad temporal.

## Trabajo realizado

- Se conservó la fábrica `crear_mlp()` existente, basada en `MLPClassifier(random_state=42, max_iter=1000)`.
- Se añadió a `ajuste_parametros.py` una búsqueda de 32 combinaciones por dataset con cinco folds. La búsqueda usa únicamente el conjunto de entrenamiento y compara los resultados finales en el conjunto de prueba.
- Se añadió `escalabilidad.py` para medir entrenamiento y predicción con 200, 400, 800, 1600 y 3200 observaciones, promediando tres corridas y usando un lote fijo de 100 muestras para predecir.
- Se generaron tablas CSV y gráficas en `resultados/` y se redactó `reporte_parte_6_mlp.md` con el funcionamiento, el análisis de resultados y la complejidad.

## Resultados obtenidos

| Dataset | Accuracy CV por defecto | Accuracy CV ajustado | Accuracy prueba por defecto | Accuracy prueba ajustado |
|---|---:|---:|---:|---:|
| Sintético | 0.7024 | 0.7095 | 0.6833 | 0.6889 |
| Wine | 0.9840 | 0.9920 | 1.0000 | 0.9815 |

El mejor MLP usa 559 iteraciones en el dataset sintético y 32 en Wine. El ajuste mejora ligeramente la prueba sintética (una observación adicional de 180), mientras que en Wine la configuración por defecto obtiene una observación correcta más que la ajustada.

En escalabilidad se midieron 1-NN, K-NN, Naïve Bayes, SVM y MLP. Distancia Mínima quedó pendiente porque la implementación del repositorio todavía lanza `NotImplementedError`.

## Validación realizada

- Se ejecutó el ajuste MLP en los dos datasets y se guardaron resultados para 32 combinaciones por dataset.
- Se ejecutaron las mediciones de escalabilidad en cinco tamaños con tres repeticiones.
- Se revisaron las tablas y las dos gráficas generadas.
- `py_compile` y `git diff --check` no reportaron errores de sintaxis ni de formato.

## Archivos relacionados

- `ajuste_parametros.py`
- `escalabilidad.py`
- `reporte_parte_6_mlp.md`
- `resultados/ajuste_mlp.csv`
- `resultados/ajuste_mlp_rejilla.csv`
- `resultados/ajuste_mlp_curvas_perdida.csv`
- `resultados/escalabilidad_tiempos.csv`
- `resultados/graficas/mlp_curva_perdida.png`
- `resultados/graficas/escalabilidad_tiempos.png`
