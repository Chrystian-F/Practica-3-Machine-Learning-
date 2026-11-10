"""
datasets.py
-----------
Carga y generación de los dos datasets de la Práctica 3.

    - Benchmark: Wine (178 muestras, 13 rasgos numéricos reales, 3 clases).
    - Sintético: generado con make_classification, con ruido estructurado.

REGLA DEL EQUIPO: nadie carga datos por su cuenta. Todos usan estas funciones
para que los 6 modelos se comparen sobre exactamente los mismos datos.
"""

from sklearn.datasets import load_wine, make_classification

SEMILLA = 42


def cargar_benchmark():
    """
    Dataset de benchmark: Wine.

    Cumple el requisito de la práctica: al menos 4 rasgos numéricos reales
    (tiene 13: alcohol, ácido málico, magnesio, fenoles, etc.) y 3 clases
    (tres cultivares de vino).

    Regresa
    -------
    X : ndarray (178, 13)   rasgos
    y : ndarray (178,)      etiquetas 0, 1, 2
    """
    datos = load_wine()
    return datos.data, datos.target


def generar_sintetico(n_muestras=600, random_state=SEMILLA):
    """
    Dataset sintético con ruido estructurado.

    El ruido NO es aleatorio puro; está construido a propósito:
      - n_redundant=2      -> 2 rasgos que son combinaciones lineales de los
                              informativos (rasgos correlacionados).
      - n_clusters_per_class=2 -> cada clase está partida en 2 grupos, así la
                              frontera entre clases no es una línea simple.
      - class_sep=0.8      -> clases poco separadas: hay traslape entre ellas.
      - flip_y=0.05        -> 5 % de etiquetas cambiadas al azar (ruido en
                              las etiquetas).
      - 1 rasgo de puro ruido (6 - 3 informativos - 2 redundantes = 1).

    Parámetros
    ----------
    n_muestras : int
        Número de muestras. Por defecto 600. El rol 6 lo cambia para medir
        cómo crecen los tiempos con n (prueba de escalabilidad).
    random_state : int
        Semilla para que todos generen exactamente los mismos datos.

    Regresa
    -------
    X : ndarray (n_muestras, 6)
    y : ndarray (n_muestras,)   etiquetas 0, 1, 2
    """
    X, y = make_classification(
        n_samples=n_muestras,
        n_features=6,
        n_informative=3,
        n_redundant=2,
        n_repeated=0,
        n_classes=3,
        n_clusters_per_class=2,
        class_sep=0.8,
        flip_y=0.05,
        random_state=random_state,
    )
    return X, y


# Diccionario que usa main.py para recorrer los datasets.
DATASETS = {
    "Sintético": generar_sintetico,
    "Wine": cargar_benchmark,
}


if __name__ == "__main__":
    # Prueba rápida: python datasets.py
    import numpy as np

    for nombre, cargar in DATASETS.items():
        X, y = cargar()
        clases, conteos = np.unique(y, return_counts=True)
        print(f"{nombre:10s} -> X {X.shape}, clases {dict(zip(clases.tolist(), conteos.tolist()))}")
