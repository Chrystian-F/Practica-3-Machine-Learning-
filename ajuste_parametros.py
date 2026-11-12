"""
ajuste_parametros.py
--------------------
Búsqueda de mejores hiperparámetros para K-NN (Rol 3), SVM (Rol 5) y MLP (Rol 6).

Procedimiento común (para que los resultados sean comparables):
  1. Dividir 70/30 con la MISMA semilla que evaluacion.py.
  2. GridSearchCV(cv=5) SOLO sobre el 70 % de entrenamiento.
  3. Evaluar el mejor modelo encontrado en el 30 % de prueba y compararlo
     contra el modelo con valores por defecto.

Los nombres de parámetros llevan el prefijo del Pipeline: 'modelo__<parametro>'.
"""

import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split

from datasets import DATASETS
from evaluacion import SEMILLA, crear_pipeline, entrenar_y_probar


def ajustar(nombre, modelo_default, rejilla, cv=5):
    """
    Corre GridSearchCV para un modelo en ambos datasets e imprime la
    comparación default vs ajustado. Regresa una lista de diccionarios.
    """
    filas = []
    for nombre_ds, cargar in DATASETS.items():
        X, y = cargar()
        X_ent, X_prueba, y_ent, y_prueba = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=SEMILLA)

        busqueda = GridSearchCV(crear_pipeline(modelo_default), rejilla,
                                cv=cv, scoring="accuracy", n_jobs=-1)
        busqueda.fit(X_ent, y_ent)

        clases = sorted(set(y))
        r_default = entrenar_y_probar(modelo_default, X_ent, y_ent, X_prueba, y_prueba, clases)
        mejor = busqueda.best_estimator_.named_steps["modelo"]
        r_ajustado = entrenar_y_probar(mejor, X_ent, y_ent, X_prueba, y_prueba, clases)

        fila = {
            "modelo": nombre,
            "dataset": nombre_ds,
            "mejores_parametros": busqueda.best_params_,
            "accuracy_cv_mejor": busqueda.best_score_,
            "accuracy_prueba_default": r_default["accuracy"],
            "accuracy_prueba_ajustado": r_ajustado["accuracy"],
        }
        fila["mejora"] = fila["accuracy_prueba_ajustado"] - fila["accuracy_prueba_default"]
        filas.append(fila)
        print(f"{nombre} | {nombre_ds}: {busqueda.best_params_} | "
              f"default={fila['accuracy_prueba_default']:.4f} "
              f"ajustado={fila['accuracy_prueba_ajustado']:.4f} "
              f"mejora={fila['mejora']:+.4f}")
    return filas


# ---------------------------------------------------------------------------
# Rol 3 — K-NN: ajuste, tabla y gráfica accuracy vs k
# ---------------------------------------------------------------------------
REJILLA_KNN = {
    "modelo__k": [1, 3, 5, 7, 9, 11, 15],
    "modelo__metrica": ["euclidiana", "manhattan"],
    "modelo__ponderado": [False, True],
}


def ajustar_knn(carpeta_resultados="resultados", cv=5):
    """
    Ajuste de K-NN en ambos datasets (mismo procedimiento que `ajustar`):
    GridSearchCV(cv=5) solo sobre el 70 % de entrenamiento y evaluación final
    en el 30 % de prueba, comparando contra KNN(k=5) con valores por defecto.

    Guarda:
      resultados/ajuste_knn.csv                  tabla defecto vs mejor
      resultados/ajuste_knn_rejilla.csv          accuracy CV de las 28 combinaciones
      resultados/graficas/knn_accuracy_vs_k.png  accuracy (CV) contra k
    """
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from modelos import KNN

    carpeta = Path(carpeta_resultados)
    (carpeta / "graficas").mkdir(parents=True, exist_ok=True)

    filas, rejillas = [], []
    for nombre_ds, cargar in DATASETS.items():
        X, y = cargar()
        X_ent, X_prueba, y_ent, y_prueba = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=SEMILLA)
        clases = sorted(set(y))

        busqueda = GridSearchCV(crear_pipeline(KNN(k=5)), REJILLA_KNN,
                                cv=cv, scoring="accuracy", n_jobs=-1)
        busqueda.fit(X_ent, y_ent)

        defecto = KNN(k=5)  # valores por defecto: k=5, euclidiana, sin ponderar
        mejor = busqueda.best_estimator_.named_steps["modelo"]
        r_def = entrenar_y_probar(defecto, X_ent, y_ent, X_prueba, y_prueba, clases)
        r_mej = entrenar_y_probar(mejor, X_ent, y_ent, X_prueba, y_prueba, clases)

        # accuracy CV de la configuración por defecto (para comparar manzanas con manzanas)
        res = pd.DataFrame(busqueda.cv_results_)
        es_def = ((res["param_modelo__k"] == 5)
                  & (res["param_modelo__metrica"] == "euclidiana")
                  & (res["param_modelo__ponderado"] == False))  # noqa: E712
        cv_def = float(res.loc[es_def, "mean_test_score"].iloc[0])

        filas.append({
            "dataset": nombre_ds,
            "parametros_defecto": "k=5, euclidiana, sin ponderar",
            "mejores_parametros": (f"k={busqueda.best_params_['modelo__k']}, "
                                   f"{busqueda.best_params_['modelo__metrica']}, "
                                   f"{'ponderado' if busqueda.best_params_['modelo__ponderado'] else 'sin ponderar'}"),
            "accuracy_cv_defecto": cv_def,
            "accuracy_cv_mejor": busqueda.best_score_,
            "accuracy_prueba_defecto": r_def["accuracy"],
            "accuracy_prueba_mejor": r_mej["accuracy"],
            "diferencia_prueba": r_mej["accuracy"] - r_def["accuracy"],
        })
        res = res.assign(dataset=nombre_ds)
        rejillas.append(res[["dataset", "param_modelo__k", "param_modelo__metrica",
                             "param_modelo__ponderado", "mean_test_score", "std_test_score"]]
                        .rename(columns=lambda c: c.replace("param_modelo__", "")))
        print(f"K-NN | {nombre_ds}: mejor = {filas[-1]['mejores_parametros']} | "
              f"CV defecto={cv_def:.4f} CV mejor={busqueda.best_score_:.4f} | "
              f"prueba defecto={r_def['accuracy']:.4f} ajustado={r_mej['accuracy']:.4f} "
              f"({filas[-1]['diferencia_prueba']:+.4f})")

    tabla = pd.DataFrame(filas)
    tabla.to_csv(carpeta / "ajuste_knn.csv", index=False, encoding="utf-8")
    rejilla = pd.concat(rejillas, ignore_index=True)
    rejilla.to_csv(carpeta / "ajuste_knn_rejilla.csv", index=False, encoding="utf-8")

    # Gráfica: accuracy (promedio de los 5 folds) contra k, una curva por
    # métrica x ponderado, un panel por dataset.
    fig, ejes = plt.subplots(1, len(DATASETS), figsize=(12, 4.5), sharey=False)
    estilos = {("euclidiana", False): ("tab:blue", "-", "o"),
               ("euclidiana", True): ("tab:blue", "--", "s"),
               ("manhattan", False): ("tab:orange", "-", "o"),
               ("manhattan", True): ("tab:orange", "--", "s")}
    for eje, nombre_ds in zip(np.atleast_1d(ejes), DATASETS):
        sub = rejilla[rejilla["dataset"] == nombre_ds]
        for (metrica, pond), (color, ls, marca) in estilos.items():
            s = sub[(sub["metrica"] == metrica) & (sub["ponderado"] == pond)].sort_values("k")
            eje.plot(s["k"], s["mean_test_score"], color=color, linestyle=ls, marker=marca,
                     label=f"{metrica}, {'ponderado' if pond else 'sin ponderar'}")
        eje.set_title(nombre_ds)
        eje.set_xlabel("k (número de vecinos)")
        eje.set_ylabel("Accuracy (validación cruzada, 5 folds)")
        eje.set_xticks(REJILLA_KNN["modelo__k"])
        eje.grid(alpha=0.3)
    np.atleast_1d(ejes)[0].legend(fontsize=8)
    fig.suptitle("K-NN: accuracy contra k (solo datos de entrenamiento)")
    fig.tight_layout()
    fig.savefig(carpeta / "graficas" / "knn_accuracy_vs_k.png", dpi=150)
    plt.close(fig)
    return tabla


if __name__ == "__main__":
    # Rol 3 — K-NN
    ajustar_knn()

    # TODO Rol 5 — SVM
    #   from modelos import crear_svm
    #   ajustar("SVM", crear_svm(), [...])

    # TODO Rol 6 — MLP
    #   from modelos import crear_mlp
    #   ajustar("MLP", crear_mlp(), {...})
    pass
