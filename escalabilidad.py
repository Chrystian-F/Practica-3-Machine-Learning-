"""Medición empírica de escalabilidad para los seis clasificadores.

Entrena con datasets sintéticos de 200, 400, 800, 1600 y 3200 muestras.
El tiempo de entrenamiento incluye el StandardScaler del Pipeline común. La
predicción usa siempre el mismo lote de 100 observaciones, de modo que los
cambios al crecer n reflejen principalmente el tamaño del conjunto entrenado.

Uso:
    python escalabilidad.py
    python escalabilidad.py --repeticiones 5

Un modelo todavía no implementado queda registrado como pendiente; el resto
de la corrida continúa para que se pueda repetir cuando el equipo lo integre.
"""

import argparse
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datasets import generar_sintetico
from evaluacion import SEMILLA, crear_pipeline
from modelos import DistanciaMinima, KNN, crear_mlp, crear_naive_bayes, crear_svm


TAMANOS = (200, 400, 800, 1600, 3200)
TAMANO_LOTE_PREDICCION = 100

# Complejidades reportadas por el equipo para interpretar las curvas.
COMPLEJIDAD = {
    "Distancia Mínima": {
        "entrenamiento_big_o": "O(n·d)",
        "prediccion_big_o_por_muestra": "O(c·d)",
        "memoria": "O(c·d)",
        "interpretacion": "Los centroides resumen cada clase; predecir no recorre n.",
    },
    "1-NN": {
        "entrenamiento_big_o": "O(1) más almacenar O(n·d)",
        "prediccion_big_o_por_muestra": "O(n·d)",
        "memoria": "O(n·d)",
        "interpretacion": "Memoriza todos los ejemplos y calcula distancias al predecir.",
    },
    "K-NN": {
        "entrenamiento_big_o": "O(1) más almacenar O(n·d)",
        "prediccion_big_o_por_muestra": "O(n·d)",
        "memoria": "O(n·d)",
        "interpretacion": "Igual que 1-NN en costo asintótico; cambia k y la votación.",
    },
    "Naïve Bayes": {
        "entrenamiento_big_o": "O(n·d)",
        "prediccion_big_o_por_muestra": "O(c·d)",
        "memoria": "O(c·d)",
        "interpretacion": "Resume medias y varianzas por clase y rasgo.",
    },
    "SVM": {
        "entrenamiento_big_o": "O(n²·d) a O(n³), según solver y datos",
        "prediccion_big_o_por_muestra": "O(s·d)",
        "memoria": "O(s·d)",
        "interpretacion": "s es el número de vectores de soporte.",
    },
    "MLP": {
        "entrenamiento_big_o": "O(épocas·n·W)",
        "prediccion_big_o_por_muestra": "O(W)",
        "memoria": "O(W)",
        "interpretacion": "W es el total de pesos; el costo depende de las épocas y la arquitectura.",
    },
}


def crear_modelos():
    """Crea una instancia nueva por combinación modelo/tamaño/repetición."""
    return {
        "Distancia Mínima": DistanciaMinima,
        "1-NN": lambda: KNN(k=1),
        "K-NN": lambda: KNN(k=5),
        "Naïve Bayes": crear_naive_bayes,
        "SVM": crear_svm,
        "MLP": crear_mlp,
    }


def medir_escalabilidad(carpeta_resultados="resultados", repeticiones=3):
    """Mide y guarda tiempos medios y desviaciones por modelo y tamaño n."""
    if repeticiones < 1:
        raise ValueError("repeticiones debe ser al menos 1")

    carpeta = Path(carpeta_resultados)
    graficas = carpeta / "graficas"
    graficas.mkdir(parents=True, exist_ok=True)

    X_pred, _ = generar_sintetico(
        n_muestras=TAMANO_LOTE_PREDICCION, random_state=SEMILLA + 1)
    filas = []
    constructores = crear_modelos()

    for n in TAMANOS:
        X, y = generar_sintetico(n_muestras=n, random_state=SEMILLA)
        print(f"\n=== Escalabilidad: n={n}, d={X.shape[1]} ===")

        for nombre, crear_modelo in constructores.items():
            tiempos_entrenamiento, tiempos_prediccion = [], []
            error = None

            for repeticion in range(repeticiones):
                pipeline = crear_pipeline(crear_modelo())
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        inicio = time.perf_counter()
                        pipeline.fit(X, y)
                        tiempos_entrenamiento.append(time.perf_counter() - inicio)

                    inicio = time.perf_counter()
                    pipeline.predict(X_pred)
                    tiempos_prediccion.append(time.perf_counter() - inicio)
                except NotImplementedError as exc:
                    error = str(exc)
                    break

            info = COMPLEJIDAD[nombre]
            fila = {
                "modelo": nombre,
                "n": n,
                "d": X.shape[1],
                "repeticiones": repeticiones,
                "muestras_lote_prediccion": len(X_pred),
                "tiempo_entrenamiento_promedio_s": (
                    float(np.mean(tiempos_entrenamiento)) if tiempos_entrenamiento else np.nan),
                "tiempo_entrenamiento_desviacion_s": (
                    float(np.std(tiempos_entrenamiento)) if tiempos_entrenamiento else np.nan),
                "tiempo_prediccion_promedio_s": (
                    float(np.mean(tiempos_prediccion)) if tiempos_prediccion else np.nan),
                "tiempo_prediccion_desviacion_s": (
                    float(np.std(tiempos_prediccion)) if tiempos_prediccion else np.nan),
                "entrenamiento_big_o": info["entrenamiento_big_o"],
                "prediccion_big_o_por_muestra": info["prediccion_big_o_por_muestra"],
                "memoria_teorica": info["memoria"],
                "interpretacion": info["interpretacion"],
                "estado": "pendiente" if error else "medido",
                "detalle": error or "",
            }
            filas.append(fila)
            if error:
                print(f"  {nombre}: pendiente ({error})")
            else:
                print(
                    f"  {nombre:17s} | entrenamiento="
                    f"{fila['tiempo_entrenamiento_promedio_s']:.6f}s | "
                    f"predicción/{len(X_pred)}="
                    f"{fila['tiempo_prediccion_promedio_s']:.6f}s"
                )

    tabla = pd.DataFrame(filas)
    ruta_csv = carpeta / "escalabilidad_tiempos.csv"
    tabla.to_csv(ruta_csv, index=False, encoding="utf-8-sig")
    _graficar(tabla, graficas / "escalabilidad_tiempos.png")
    print(f"\nTabla: {ruta_csv}")
    print(f"Gráfica: {graficas / 'escalabilidad_tiempos.png'}")
    return tabla


def _graficar(tabla, ruta):
    """Dibuja entrenamiento y predicción contra n en escala logarítmica."""
    disponibles = tabla[tabla["estado"] == "medido"]
    fig, ejes = plt.subplots(1, 2, figsize=(13, 5))

    for modelo, datos in disponibles.groupby("modelo", sort=False):
        datos = datos.sort_values("n")
        ejes[0].plot(
            datos["n"], datos["tiempo_entrenamiento_promedio_s"],
            marker="o", label=modelo)
        ejes[1].plot(
            datos["n"], datos["tiempo_prediccion_promedio_s"],
            marker="o", label=modelo)

    for eje, titulo, ylabel in (
        (ejes[0], "Tiempo de entrenamiento", "Segundos (promedio)"),
        (ejes[1], f"Tiempo de predicción ({TAMANO_LOTE_PREDICCION} muestras)",
         "Segundos (promedio)"),
    ):
        eje.set_title(titulo)
        eje.set_xlabel("Número de muestras de entrenamiento (n)")
        eje.set_ylabel(ylabel)
        eje.set_yscale("log")
        eje.set_xticks(TAMANOS)
        eje.grid(True, which="both", alpha=0.3)
        eje.legend(fontsize=8)

    fig.suptitle("Escalabilidad: tiempos medidos frente al tamaño del dataset")
    fig.tight_layout()
    fig.savefig(ruta, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeticiones", type=int, default=3)
    args = parser.parse_args()
    medir_escalabilidad(repeticiones=args.repeticiones)
