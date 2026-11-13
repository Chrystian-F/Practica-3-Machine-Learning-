"""
ajuste_naive_bayes.py
---------------------
Rol 4: prueba extra de var_smoothing en Naïve Bayes gaussiano.

GaussianNB casi no tiene hiperparámetros; solo var_smoothing, que suma a todas
las varianzas una fracción de la varianza máxima de los rasgos (estabilidad
numérica). Se prueba var_smoothing ∈ {1e-9, 1e-7, 1e-5} con la evaluación común
(mismas particiones y Pipeline con escalado dentro del fold que los demás modelos).

Guarda: resultados/ajuste_naive_bayes.csv

Uso:
    python ajuste_naive_bayes.py
"""

from pathlib import Path

import pandas as pd
from sklearn.naive_bayes import GaussianNB

from datasets import DATASETS
from evaluacion import evaluar_holdout, evaluar_kfold

VALORES_VAR_SMOOTHING = [1e-9, 1e-7, 1e-5]  # 1e-9 es el valor por defecto


def probar_var_smoothing(carpeta_resultados="resultados", k=10):
    filas = []
    for nombre_ds, cargar in DATASETS.items():
        X, y = cargar()
        for vs in VALORES_VAR_SMOOTHING:
            ho = evaluar_holdout(GaussianNB(var_smoothing=vs), X, y)
            kf = evaluar_kfold(GaussianNB(var_smoothing=vs), X, y, k=k)
            filas.append({
                "dataset": nombre_ds,
                "var_smoothing": vs,
                "accuracy_holdout": ho["accuracy"],
                "accuracy_kfold": kf["accuracy"],
                "accuracy_kfold_std": kf["accuracy_std"],
            })
            print(f"Naïve Bayes | {nombre_ds:9s} | var_smoothing={vs:g} | "
                  f"Hold-Out={ho['accuracy']:.4f} | "
                  f"K-Fold={kf['accuracy']:.4f} ± {kf['accuracy_std']:.4f}")

    tabla = pd.DataFrame(filas)
    carpeta = Path(carpeta_resultados)
    carpeta.mkdir(exist_ok=True)
    tabla.to_csv(carpeta / "ajuste_naive_bayes.csv", index=False, encoding="utf-8")
    return tabla


if __name__ == "__main__":
    probar_var_smoothing()
