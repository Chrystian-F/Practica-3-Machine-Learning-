
"""
graficas.py
------------
RESPONSABLE: Rol 2.

Genera las gráficas de evaluación de los 6 clasificadores:
- Matrices de confusión (Hold-Out) para ambos datasets.
- Comparación de accuracy: Hold-Out vs K-Fold.
- Tiempos de entrenamiento y predicción con escala logarítmica.
- Fronteras de decisión de los 6 modelos en el dataset sintético,
  reducido a 2 dimensiones mediante PCA.

Los resultados se guardan en resultados/graficas/.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

from datasets import DATASETS, generar_sintetico
from modelos import (
    DistanciaMinima,
    KNN,
    crear_mlp,
    crear_naive_bayes,
    crear_svm,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).parent
CARPETA_RESULTADOS = BASE_DIR / "resultados"
CARPETA_GRAFICAS = CARPETA_RESULTADOS / "graficas"

ARCHIVO_CSV = CARPETA_RESULTADOS / "resultados.csv"
ARCHIVO_JSON = CARPETA_RESULTADOS / "matrices_confusion.json"

MODELOS = {
    "Distancia Mínima": lambda: DistanciaMinima(),
    "1-NN": lambda: KNN(k=1),
    "K-NN": lambda: KNN(k=5),
    "Naïve Bayes": crear_naive_bayes,
    "SVM": crear_svm,
    "MLP": crear_mlp,
}

VALIDACIONES = ["Hold-Out 70/30", "K-Fold (k=10)"]


# ============================================================
# UTILIDADES
# ============================================================

def preparar_directorio():
    """Crea la carpeta de gráficas si todavía no existe."""
    CARPETA_GRAFICAS.mkdir(parents=True, exist_ok=True)


def cargar_resultados():
    """Carga los resultados generados por main.py."""
    if not ARCHIVO_CSV.exists():
        raise FileNotFoundError(
            "No se encontró resultados/resultados.csv. "
            "Primero ejecuta: python main.py"
        )

    if not ARCHIVO_JSON.exists():
        raise FileNotFoundError(
            "No se encontró resultados/matrices_confusion.json. "
            "Primero ejecuta: python main.py"
        )

    df = pd.read_csv(ARCHIVO_CSV)

    with open(ARCHIVO_JSON, "r", encoding="utf-8") as archivo:
        matrices = json.load(archivo)

    return df, matrices


def guardar_figura(nombre):
    """Guarda la figura actual como PNG."""
    ruta = CARPETA_GRAFICAS / nombre
    plt.savefig(ruta, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {ruta.name}")


# ============================================================
# 1. MATRICES DE CONFUSIÓN
# ============================================================

def graficar_matrices_confusion(matrices):
    """
    Genera una matriz de confusión por modelo y dataset,
    utilizando exclusivamente los resultados Hold-Out.
    """
    print("\n[1/4] Generando matrices de confusión...")

    for dataset in DATASETS:
        for modelo in MODELOS:

            clave = f"{dataset} | {modelo} | Hold-Out 70/30"

            if clave not in matrices:
                raise KeyError(
                    f"No se encontró la matriz: {clave}"
                )

            matriz = np.array(
                matrices[clave]["matriz_confusion"]
            )

            fig, ax = plt.subplots(figsize=(6, 5))

            imagen = ax.imshow(matriz, cmap="Blues")

            ax.set_title(
                f"Matriz de confusión\n{modelo} - {dataset}"
            )
            ax.set_xlabel("Clase predicha")
            ax.set_ylabel("Clase real")

            clases = np.arange(matriz.shape[0])

            ax.set_xticks(clases)
            ax.set_yticks(clases)

            for i in range(matriz.shape[0]):
                for j in range(matriz.shape[1]):
                    ax.text(
                        j,
                        i,
                        str(matriz[i, j]),
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=11,
                    )

            fig.colorbar(imagen, ax=ax, label="Muestras")

            nombre = (
                f"matriz_{dataset.lower()}_"
                f"{modelo.lower().replace(' ', '_').replace('ï', 'i')}.png"
            )

            plt.tight_layout()
            guardar_figura(nombre)


# ============================================================
# 2. COMPARACIÓN DE ACCURACY
# ============================================================

def graficar_accuracy(df):
    """
    Compara el accuracy de Hold-Out y K-Fold
    para los seis clasificadores en cada dataset.

    La desviación estándar se representa en K-Fold.
    """
    print("\n[2/4] Generando gráficas de accuracy...")

    for dataset in DATASETS:

        datos = df[df["dataset"] == dataset]

        holdout = datos[
            datos["validacion"] == "Hold-Out 70/30"
        ].set_index("modelo")

        kfold = datos[
            datos["validacion"] == "K-Fold (k=10)"
        ].set_index("modelo")

        modelos = list(MODELOS.keys())

        valores_holdout = [
            holdout.loc[m, "accuracy"] for m in modelos
        ]

        valores_kfold = [
            kfold.loc[m, "accuracy"] for m in modelos
        ]

        errores_kfold = [
            kfold.loc[m, "accuracy_std"]
            if pd.notna(kfold.loc[m, "accuracy_std"])
            else 0
            for m in modelos
        ]

        x = np.arange(len(modelos))
        ancho = 0.36

        fig, ax = plt.subplots(figsize=(11, 6))

        ax.bar(
            x - ancho / 2,
            valores_holdout,
            ancho,
            label="Hold-Out 70/30",
        )

        ax.bar(
            x + ancho / 2,
            valores_kfold,
            ancho,
            yerr=errores_kfold,
            capsize=4,
            label="K-Fold (k=10)",
        )

        ax.set_title(f"Accuracy por modelo - {dataset}")
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Clasificador")
        ax.set_xticks(x)
        ax.set_xticklabels(modelos, rotation=20, ha="right")
        ax.set_ylim(0, 1.15)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        guardar_figura(
            f"accuracy_{dataset.lower()}.png"
        )


# ============================================================
# 3. TIEMPOS DE ENTRENAMIENTO Y PREDICCIÓN
# ============================================================

def graficar_tiempos(df):
    """
    Compara los tiempos de entrenamiento y predicción
    por modelo utilizando escala logarítmica.

    Se utiliza el promedio de los dos métodos de validación
    para cada modelo y dataset.
    """
    print("\n[3/4] Generando gráficas de tiempos...")

    for dataset in DATASETS:

        datos = df[df["dataset"] == dataset]

        agrupados = datos.groupby("modelo")[
            [
                "tiempo_entrenamiento_s",
                "tiempo_prediccion_s",
            ]
        ].mean()

        modelos = list(MODELOS.keys())

        entrenamiento = [
            agrupados.loc[m, "tiempo_entrenamiento_s"]
            for m in modelos
        ]

        prediccion = [
            agrupados.loc[m, "tiempo_prediccion_s"]
            for m in modelos
        ]

        x = np.arange(len(modelos))
        ancho = 0.36

        fig, ax = plt.subplots(figsize=(11, 6))

        ax.bar(
            x - ancho / 2,
            entrenamiento,
            ancho,
            label="Entrenamiento",
        )

        ax.bar(
            x + ancho / 2,
            prediccion,
            ancho,
            label="Predicción",
        )

        ax.set_yscale("log")
        ax.set_title(f"Tiempos de ejecución - {dataset}")
        ax.set_ylabel("Tiempo (segundos, escala logarítmica)")
        ax.set_xlabel("Clasificador")
        ax.set_xticks(x)
        ax.set_xticklabels(modelos, rotation=20, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        guardar_figura(
            f"tiempos_{dataset.lower()}.png"
        )


# ============================================================
# 4. FRONTERAS DE DECISIÓN CON PCA
# ============================================================

def graficar_fronteras_decision():
    """
    Reduce el dataset sintético a dos dimensiones mediante PCA
    y representa las fronteras de decisión de los seis modelos.

    Los modelos se entrenan sobre las dos componentes principales.
    El escalado se mantiene dentro del Pipeline.
    """
    print("\n[4/4] Generando fronteras de decisión con PCA...")

    X, y = generar_sintetico()

    # Reducción de seis características a dos componentes.
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # Malla de puntos para representar las regiones de decisión.
    margen = 1.0

    x_min = X_2d[:, 0].min() - margen
    x_max = X_2d[:, 0].max() + margen
    y_min = X_2d[:, 1].min() - margen
    y_max = X_2d[:, 1].max() + margen

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 220),
        np.linspace(y_min, y_max, 220),
    )

    malla = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(
        2, 3,
        figsize=(16, 10),
    )

    axes = axes.ravel()

    for ax, (nombre, crear_modelo) in zip(
        axes, MODELOS.items()
    ):

        modelo = Pipeline([
            ("escalador", StandardScaler()),
            ("clasificador", crear_modelo()),
        ])

        modelo.fit(X_2d, y)

        predicciones = modelo.predict(malla)
        regiones = predicciones.reshape(xx.shape)

        ax.contourf(
            xx,
            yy,
            regiones,
            alpha=0.30,
            levels=np.arange(len(np.unique(y)) + 1) - 0.5,
        )

        puntos = ax.scatter(
            X_2d[:, 0],
            X_2d[:, 1],
            c=y,
            edgecolors="black",
            linewidths=0.3,
            s=20,
        )

        ax.set_title(nombre)
        ax.set_xlabel("Componente principal 1")
        ax.set_ylabel("Componente principal 2")
        ax.grid(alpha=0.2)

    fig.suptitle(
        "Fronteras de decisión de los seis clasificadores "
        "sobre el dataset sintético reducido mediante PCA",
        fontsize=14,
    )

    plt.tight_layout()
    guardar_figura("fronteras_decision_pca.png")


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    print("=" * 60)
    print("GENERACIÓN DE GRÁFICAS - PRÁCTICA 3")
    print("ROL 2: DISTANCIA MÍNIMA")
    print("=" * 60)

    preparar_directorio()

    df, matrices = cargar_resultados()

    graficar_matrices_confusion(matrices)
    graficar_accuracy(df)
    graficar_tiempos(df)
    graficar_fronteras_decision()

    print("\n" + "=" * 60)
    print("GRÁFICAS GENERADAS CORRECTAMENTE")
    print(f"Ubicación: {CARPETA_GRAFICAS}")
    print("=" * 60)


if __name__ == "__main__":
    main()
