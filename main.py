"""
main.py
-------
Corre los 24 experimentos de la práctica:

    6 modelos  x  2 datasets  x  2 métodos de validación

y guarda:
    resultados/resultados.csv            tabla con métricas y tiempos
    resultados/matrices_confusion.json   matrices de confusión (para el Rol 2)

Si un modelo todavía no está implementado (lanza NotImplementedError),
se salta con un aviso y el resto sigue corriendo.

Uso:
    python main.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from datasets import DATASETS
from evaluacion import evaluar_holdout, evaluar_kfold
from modelos import KNN, DistanciaMinima, crear_mlp, crear_naive_bayes, crear_svm

K_FOLDS = 10
CARPETA_RESULTADOS = Path(__file__).parent / "resultados"

# Cada entrada es una función que crea un modelo NUEVO sin entrenar.
MODELOS = {
    "Distancia Mínima": lambda: DistanciaMinima(),
    "1-NN": lambda: KNN(k=1),
    "K-NN": lambda: KNN(k=5),
    "Naïve Bayes": crear_naive_bayes,
    "SVM": crear_svm,
    "MLP": crear_mlp,
}

VALIDACIONES = {
    "Hold-Out 70/30": lambda modelo, X, y: evaluar_holdout(modelo, X, y),
    f"K-Fold (k={K_FOLDS})": lambda modelo, X, y: evaluar_kfold(modelo, X, y, k=K_FOLDS),
}


def main():
    CARPETA_RESULTADOS.mkdir(exist_ok=True)
    filas = []
    matrices = {}
    pendientes = set()

    for nombre_ds, cargar in DATASETS.items():
        X, y = cargar()
        print(f"\n=== Dataset: {nombre_ds}  (n={X.shape[0]}, d={X.shape[1]}) ===")

        for nombre_modelo, crear in MODELOS.items():
            for nombre_val, validar in VALIDACIONES.items():
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        r = validar(crear(), X, y)
                except NotImplementedError:
                    pendientes.add(nombre_modelo)
                    print(f"  [PENDIENTE] {nombre_modelo:17s} | {nombre_val}")
                    continue

                filas.append({
                    "dataset": nombre_ds,
                    "modelo": nombre_modelo,
                    "validacion": nombre_val,
                    "accuracy": r["accuracy"],
                    "accuracy_std": r.get("accuracy_std", np.nan),
                    "precision": r["precision"],
                    "recall": r["recall"],
                    "f1": r["f1"],
                    "tiempo_entrenamiento_s": r["tiempo_entrenamiento_s"],
                    "tiempo_prediccion_s": r["tiempo_prediccion_s"],
                })
                matrices[f"{nombre_ds} | {nombre_modelo} | {nombre_val}"] = {
                    "matriz_confusion": r["matriz_confusion"].tolist(),
                    "sensibilidad_por_clase": np.round(r["sensibilidad_por_clase"], 4).tolist(),
                    "especificidad_por_clase": np.round(r["especificidad_por_clase"], 4).tolist(),
                }
                std = f" ± {r['accuracy_std']:.4f}" if "accuracy_std" in r else ""
                print(f"  {nombre_modelo:17s} | {nombre_val:15s} | accuracy = {r['accuracy']:.4f}{std}")

    df = pd.DataFrame(filas)
    ruta_csv = CARPETA_RESULTADOS / "resultados.csv"
    df.to_csv(ruta_csv, index=False, encoding="utf-8")
    with open(CARPETA_RESULTADOS / "matrices_confusion.json", "w", encoding="utf-8") as f:
        json.dump(matrices, f, ensure_ascii=False, indent=2)

    print(f"\nExperimentos completados: {len(df)} de {len(MODELOS) * len(DATASETS) * len(VALIDACIONES)}")
    if pendientes:
        print("Modelos pendientes de implementar:", ", ".join(sorted(pendientes)))
    print(f"Resultados guardados en {ruta_csv}")

    if not df.empty:
        print("\nAccuracy por modelo:")
        tabla = df.pivot_table(index="modelo", columns=["dataset", "validacion"], values="accuracy")
        print(tabla.round(4).to_string())


if __name__ == "__main__":
    main()
