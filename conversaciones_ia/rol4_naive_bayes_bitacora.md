# Bitácora de apoyo con IA — Rol 4: Naïve Bayes y reporte

**Fecha:** 13 de noviembre de 2026  
**Proyecto:** Práctica 3 de Machine Learning  
**Responsabilidad:** Naïve Bayes, prueba de `var_smoothing`


## Solicitud y alcance

El estudiante compartió el plan del el Rol 4. Se identificaron estos entregables: la función `crear_naive_bayes()`, la prueba de `var_smoothing` ∈ {1e-9, 1e-7, 1e-5}, el apartado teórico de Naïve Bayes (Bayes, independencia condicional, versión gaussiana, relación con el clasificador euclidiano y complejidad), la respuesta a "¿cómo encontrar los mejores parámetros para cada dataset?".

## Trabajo realizado

- Se añadió `ajuste_naive_bayes.py`, que prueba `var_smoothing` con `evaluar_holdout` y `evaluar_kfold` del repositorio (mismas particiones y escalado dentro del fold que los demás modelos) y guarda `resultados/ajuste_naive_bayes.csv`.
- Se redactó `resultados/seccion_reporte_rol4_naive_bayes.md` con el funcionamiento, el análisis de resultados, el ajuste de parámetros, la respuesta sobre Grid Search y Random Search con validación cruzada, y la complejidad temporal.
- Se dieron instrucciones de PowerShell y Git para clonar, crear la rama, instalar dependencias, ejecutar y subir los cambios.

## Resultados obtenidos

| Dataset | Accuracy Hold-Out | Accuracy K-Fold (k = 10) |
|---|---|---|
| Sintético | 0.5833 | 0.6083 ± 0.0611 |
| Wine | 1.0000 | 0.9778 ± 0.0369 |

`var_smoothing` ∈ {1e-9, 1e-7, 1e-5} produce exactamente el mismo resultado en los dos datasets, porque con los datos estandarizados las varianzas están cerca de 1 y ese término es despreciable. Se conserva el valor por defecto. En el dataset sintético, Naïve Bayes obtiene el resultado más bajo de los modelos con ajuste; en Wine, de los más altos.

## Validación realizada

- Se ejecutó `main.py` con el repositorio y se obtuvieron los 24 experimentos (6 modelos × 2 datasets × 2 validaciones).
- Se ejecutó `ajuste_naive_bayes.py` en el equipo del estudiante y los números coincidieron con los esperados.
- Se revisó con `git status` que solo se agregaran los archivos del Rol 4 (sin el entorno virtual) y se subió la rama `modelo-naive-bayes`.

## Archivos relacionados

- `ajuste_naive_bayes.py`
- `resultados/ajuste_naive_bayes.csv`
- `resultados/seccion_reporte_rol4_naive_bayes.md`


## Lo que revisó o modificó el estudiante
