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


if __name__ == "__main__":
    # TODO Rol 3 — K-NN
    #   from modelos import KNN
    #   ajustar("K-NN", KNN(k=5), {
    #       "modelo__k": [1, 3, 5, 7, 9, 11, 15],
    #       "modelo__metrica": ["euclidiana", "manhattan"],
    #       "modelo__ponderado": [False, True],
    #   })

    # TODO Rol 5 — SVM
    #   from modelos import crear_svm
    #   ajustar("SVM", crear_svm(), [...])

    # TODO Rol 6 — MLP
    #   from modelos import crear_mlp
    #   ajustar("MLP", crear_mlp(), {...})
    pass
