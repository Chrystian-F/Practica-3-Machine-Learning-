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


# ---------------------------------------------------------------------------
# Rol 5 — SVM: ajuste, tabla y mapa de calor de C contra gamma (kernel RBF)
# ---------------------------------------------------------------------------
SVM_VALORES_C = [0.1, 1, 10, 100]
SVM_VALORES_GAMMA = ["scale", 0.01, 0.1, 1]

# Una rejilla por kernel para no entrenar combinaciones repetidas: el kernel
# lineal no usa gamma ni degree, y el RBF no usa degree.
# Total: 4 (lineal) + 16 (RBF) + 32 (polinomial) = 52 combinaciones.
REJILLA_SVM = [
    {"modelo__kernel": ["linear"], "modelo__C": SVM_VALORES_C},
    {"modelo__kernel": ["rbf"], "modelo__C": SVM_VALORES_C,
     "modelo__gamma": SVM_VALORES_GAMMA},
    {"modelo__kernel": ["poly"], "modelo__C": SVM_VALORES_C,
     "modelo__gamma": SVM_VALORES_GAMMA, "modelo__degree": [2, 3]},
]


def _parametros_svm(svc):
    """Parámetros que usa el kernel de un SVC, como texto:
    'kernel=rbf, C=1, gamma=scale'."""
    partes = [f"kernel={svc.kernel}", f"C={svc.C:g}"]
    if svc.kernel != "linear":
        partes.append(f"gamma={svc.gamma}")
    if svc.kernel == "poly":
        partes.append(f"degree={svc.degree}")
    return ", ".join(partes)


def ajustar_svm(carpeta_resultados="resultados", cv=5):
    """
    Ajuste de SVM en ambos datasets (mismo procedimiento que `ajustar`):
    GridSearchCV(cv=5) solo sobre el 70 % de entrenamiento y evaluación final
    en el 30 % de prueba, comparando contra crear_svm() (valores por defecto:
    kernel RBF, C = 1, gamma = "scale").

    Si varias combinaciones empatan en accuracy, GridSearchCV se queda con la
    primera de la rejilla; la tabla dice cuántas empataron.

    Guarda:
      resultados/ajuste_svm.csv                       tabla defecto vs mejor
      resultados/ajuste_svm_rejilla.csv               accuracy CV de las 52 combinaciones
      resultados/graficas/svm_mapa_calor_C_gamma.png  accuracy (CV) de C contra gamma, kernel RBF
    """
    from pathlib import Path

    import pandas as pd
    from sklearn.model_selection import cross_val_score

    from modelos import crear_svm

    carpeta = Path(carpeta_resultados)
    (carpeta / "graficas").mkdir(parents=True, exist_ok=True)

    print("SVM: buscando en la rejilla. En el dataset sintético algunas combinaciones "
          "polinomiales (C y gamma grandes) tardan; puede tomar unos minutos...")
    defecto = crear_svm()
    filas, rejillas, paneles = [], [], []
    for nombre_ds, cargar in DATASETS.items():
        X, y = cargar()
        X_ent, X_prueba, y_ent, y_prueba = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=SEMILLA)
        clases = sorted(set(y))

        busqueda = GridSearchCV(crear_pipeline(defecto), REJILLA_SVM, cv=cv,
                                scoring="accuracy", n_jobs=-1, return_train_score=True)
        busqueda.fit(X_ent, y_ent)
        res = busqueda.cv_results_
        mejor = busqueda.best_estimator_.named_steps["modelo"]  # ya entrenado con el 70 %
        empatadas = int((res["rank_test_score"] == 1).sum())

        # Accuracy de validación cruzada de la configuración por defecto, con
        # los mismos folds que GridSearchCV (para comparar manzanas con manzanas).
        cv_def = cross_val_score(crear_pipeline(defecto), X_ent, y_ent,
                                 cv=cv, scoring="accuracy").mean()

        # Evaluación final, una sola vez, en el 30 % de prueba.
        r_def = entrenar_y_probar(defecto, X_ent, y_ent, X_prueba, y_prueba, clases)
        r_mej = entrenar_y_probar(mejor, X_ent, y_ent, X_prueba, y_prueba, clases)

        # Vectores de soporte: son la "s" del costo de predicción O(s·d).
        svc_def = crear_pipeline(defecto).fit(X_ent, y_ent).named_steps["modelo"]

        filas.append({
            "dataset": nombre_ds,
            "parametros_defecto": _parametros_svm(defecto),
            "mejores_parametros": _parametros_svm(mejor),
            "accuracy_cv_defecto": cv_def,
            "accuracy_cv_mejor": busqueda.best_score_,
            "accuracy_prueba_defecto": r_def["accuracy"],
            "accuracy_prueba_mejor": r_mej["accuracy"],
            "diferencia_prueba": r_mej["accuracy"] - r_def["accuracy"],
            "diferencia_cv": busqueda.best_score_ - cv_def,
            "combinaciones_empatadas": empatadas,
            "n_entrenamiento": len(y_ent),
            "vectores_soporte_defecto": int(svc_def.n_support_.sum()),
            "vectores_soporte_mejor": int(mejor.n_support_.sum()),
        })

        # Las 52 combinaciones: accuracy de validación y de entrenamiento
        # (promedio de los folds) y tiempo promedio de un entrenamiento.
        combinaciones = pd.DataFrame(res["params"]).rename(
            columns=lambda c: c.replace("modelo__", ""))
        for columna in ("mean_test_score", "std_test_score", "mean_train_score",
                        "rank_test_score", "mean_fit_time"):
            combinaciones[columna] = res[columna]
        rejillas.append(combinaciones.assign(dataset=nombre_ds))

        paneles.append({
            "titulo": (f"{nombre_ds} (entrenamiento: n = {len(y_ent)}, d = {X.shape[1]})\n"
                       f"mejor de la rejilla: {_parametros_svm(mejor)}"
                       + (f" (empate de {empatadas})" if empatadas > 1 else "")),
            "n_rasgos": X.shape[1],
            "azar": 1 / len(clases),
            "accuracy": {(p["modelo__C"], p["modelo__gamma"]): a
                         for p, a in zip(res["params"], res["mean_test_score"])
                         if p["modelo__kernel"] == "rbf"},
            "defecto": (defecto.C, defecto.gamma) if defecto.kernel == "rbf" else None,
            "mejor": (mejor.C, mejor.gamma) if mejor.kernel == "rbf" else None,
        })
        print(f"SVM | {nombre_ds}: mejor = {filas[-1]['mejores_parametros']} | "
              f"CV defecto={cv_def:.4f} CV mejor={busqueda.best_score_:.4f} | "
              f"prueba defecto={r_def['accuracy']:.4f} ajustado={r_mej['accuracy']:.4f} "
              f"({filas[-1]['diferencia_prueba']:+.4f})")

    tabla = pd.DataFrame(filas)
    tabla.to_csv(carpeta / "ajuste_svm.csv", index=False, encoding="utf-8")
    rejilla = pd.concat(rejillas, ignore_index=True)
    rejilla["degree"] = rejilla["degree"].astype("Int64")  # entero; vacío si no es poly
    rejilla = rejilla[["dataset", "kernel", "C", "gamma", "degree", "mean_test_score",
                       "std_test_score", "mean_train_score", "rank_test_score", "mean_fit_time"]]
    rejilla.to_csv(carpeta / "ajuste_svm_rejilla.csv", index=False, encoding="utf-8")
    _mapa_calor_svm(paneles, carpeta / "graficas" / "svm_mapa_calor_C_gamma.png", cv)
    return tabla


def _mapa_calor_svm(paneles, ruta, cv):
    """
    Mapa de calor del kernel RBF: accuracy de validación cruzada para cada par
    (C, gamma), un panel por dataset y la misma escala de color en todos.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # Escala de un solo tono: más oscuro = mayor accuracy. Va del azar
    # (1 / número de clases) a 1, así los paneles se pueden comparar.
    azules = LinearSegmentedColormap.from_list("azules", [
        "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"])
    minimo = min(panel["azar"] for panel in paneles)
    fondo, tinta, tinta_suave = "#ffffff", "#0b0b0b", "#52514e"

    def luminancia(color):
        """Luminancia relativa de un color RGB: 0 = negro, 1 = blanco."""
        r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
                   for c in color[:3])
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    fig, ejes = plt.subplots(1, len(paneles), figsize=(5.6 * len(paneles) + 1.2, 5.6),
                             squeeze=False, layout="constrained", facecolor=fondo)
    for eje, panel in zip(ejes.ravel(), paneles):
        # Columnas ordenadas por el valor real de gamma. Con los datos
        # estandarizados la varianza de X es 1, así que "scale" vale 1 / d.
        escala = 1 / panel["n_rasgos"]
        gammas = sorted(SVM_VALORES_GAMMA, key=lambda g: escala if g == "scale" else g)
        matriz = [[panel["accuracy"][(c, g)] for g in gammas] for c in SVM_VALORES_C]

        # Celdas que se rotulan: configuración por defecto y la mejor.
        marcas = {}
        for nombre in ("defecto", "mejor"):
            if panel[nombre] in panel["accuracy"]:
                marcas.setdefault(panel[nombre], []).append(nombre)

        malla = eje.pcolormesh(matriz, cmap=azules, vmin=minimo, vmax=1.0,
                               edgecolors=fondo, linewidth=2)
        # Valor escrito en cada celda: texto blanco sobre las celdas oscuras
        # y negro sobre las claras.
        for i, fila in enumerate(matriz):
            for j, valor in enumerate(fila):
                marca = "\ny ".join(marcas.get((SVM_VALORES_C[i], gammas[j]), []))
                color_celda = azules((valor - minimo) / (1.0 - minimo))
                eje.text(j + 0.5, i + 0.5, f"{valor:.3f}" + (f"\n{marca}" if marca else ""),
                         ha="center", va="center", fontsize=10,
                         fontweight="bold" if marca else "normal",
                         color=fondo if luminancia(color_celda) < 0.35 else tinta)

        eje.set_xticks([j + 0.5 for j in range(len(gammas))],
                       [f"scale\n(1/d ≈ {escala:.3f})" if g == "scale" else f"{g:g}"
                        for g in gammas])
        eje.set_yticks([i + 0.5 for i in range(len(SVM_VALORES_C))],
                       [f"{c:g}" for c in SVM_VALORES_C])
        eje.set_xlabel("gamma", color=tinta_suave)
        eje.set_ylabel("C", color=tinta_suave)
        eje.set_title(panel["titulo"], fontsize=10.5, color=tinta)
        eje.tick_params(length=0, colors=tinta_suave)
        eje.set_aspect("equal")
        for borde in eje.spines.values():
            borde.set_visible(False)

    barra = fig.colorbar(malla, ax=ejes.ravel().tolist(), shrink=0.75, pad=0.03)
    barra.set_label(f"Accuracy (validación cruzada, {cv} folds)", color=tinta_suave)
    barra.ax.tick_params(length=0, colors=tinta_suave)
    barra.outline.set_visible(False)
    fig.suptitle("SVM con kernel RBF: accuracy de C contra gamma (solo datos de entrenamiento)",
                 color=tinta)
    fig.savefig(ruta, dpi=150, facecolor=fondo)
    plt.close(fig)


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

    # Rol 5 — SVM
    ajustar_svm()

    # TODO Rol 6 — MLP
    #   from modelos import crear_mlp
    #   ajustar("MLP", crear_mlp(), {...})
    pass
