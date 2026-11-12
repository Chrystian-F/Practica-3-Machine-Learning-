"""
modelos/knn.py
--------------
RESPONSABLE: Rol 3.

Clasificador K-NN (implementación DESDE CERO, solo NumPy).
Prohibido usar KNeighborsClassifier u otros clasificadores de scikit-learn.

1-NN es simplemente KNN(k=1); main.py lo registra como modelo aparte.

Se hereda de BaseEstimator y ClassifierMixin para que funcione dentro de
Pipeline, cross_val_score y GridSearchCV. En GridSearchCV los parámetros se
nombran con prefijo del Pipeline: 'modelo__k', 'modelo__metrica', ...
NO escalen los datos aquí: el escalado ya lo hace el Pipeline de evaluacion.py.

Idea del algoritmo
    fit      -> memoriza el conjunto de entrenamiento (no aprende nada).
    predict  -> para cada punto nuevo: calcula su distancia a TODOS los puntos
                de entrenamiento, toma los k más cercanos y vota.

Regla de desempate
    Si dos o más clases quedan empatadas en la votación, gana la clase cuyos
    vecinos están más cerca: el vecino más cercano vale un poco más que el
    segundo, éste más que el tercero, etc. (ver _BONO_DESEMPATE). Con k = 1 no
    hay empates posibles.

Complejidad
    fit:                 O(1) tiempo (solo guarda referencias), O(n·d) memoria.
    predict (por punto): O(n·d) para las distancias + O(n) para argpartition.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted

# Sumando este bono (pequeñísimo comparado con 1 voto) al voto de cada vecino,
# según su posición en el orden de cercanía, los empates se rompen a favor de
# la clase con los vecinos más cercanos sin alterar nunca una mayoría real.
_BONO_DESEMPATE = 1e-6

# Máximo de elementos de la matriz intermedia (n_bloque, n_ent, d) en Manhattan,
# para no gastar memoria de más cuando n crece (prueba de escalabilidad).
_MAX_ELEMENTOS_BLOQUE = 5_000_000


class KNN(BaseEstimator, ClassifierMixin):

    def __init__(self, k=5, metrica="euclidiana", ponderado=False):
        # REGLA de scikit-learn: en __init__ solo se guardan los parámetros,
        # con el mismo nombre que el argumento. Nada de cálculos aquí.
        self.k = k
        self.metrica = metrica        # "euclidiana" o "manhattan"
        self.ponderado = ponderado    # True -> voto pesado por 1/(distancia + 1e-9)

    # ------------------------------------------------------------------
    # Entrenamiento: solo memoriza
    # ------------------------------------------------------------------
    def fit(self, X, y):
        X = check_array(X, dtype=float)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X y y deben tener el mismo número de filas.")
        if not isinstance(self.k, (int, np.integer)) or self.k < 1:
            raise ValueError("k debe ser un entero >= 1.")
        if self.metrica not in ("euclidiana", "manhattan"):
            raise ValueError("metrica debe ser 'euclidiana' o 'manhattan'.")

        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)
        # Etiquetas como índices 0..c-1, para poder contar votos con una matriz.
        self._y_idx = np.searchsorted(self.classes_, y)
        return self

    # ------------------------------------------------------------------
    # Distancias (vectorizadas)
    # ------------------------------------------------------------------
    def _distancias(self, X):
        """Matriz (n_prueba, n_entrenamiento) con la distancia de cada punto
        de X a cada punto de entrenamiento, sin ciclos por punto."""
        if self.metrica == "euclidiana":
            # ||a-b||² = ||a||² + ||b||² - 2·a·b  (una sola multiplicación de matrices)
            cuad = (
                (X ** 2).sum(axis=1)[:, None]
                + (self.X_ ** 2).sum(axis=1)[None, :]
                - 2.0 * X @ self.X_.T
            )
            return np.sqrt(np.maximum(cuad, 0.0))  # maximum: evita negativos por redondeo

        # Manhattan: suma de |a-b|. Por bloques de filas para acotar la memoria.
        n_ent, d = self.X_.shape
        bloque = max(1, _MAX_ELEMENTOS_BLOQUE // max(1, n_ent * d))
        D = np.empty((X.shape[0], n_ent))
        for i in range(0, X.shape[0], bloque):
            parte = X[i:i + bloque]
            D[i:i + bloque] = np.abs(parte[:, None, :] - self.X_[None, :, :]).sum(axis=2)
        return D

    # ------------------------------------------------------------------
    # Predicción: k vecinos + votación
    # ------------------------------------------------------------------
    def predict(self, X):
        check_is_fitted(self, ["X_", "y_", "classes_"])
        X = check_array(X, dtype=float)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("X tiene un número de rasgos distinto al del entrenamiento.")

        n_ent = self.X_.shape[0]
        k = min(self.k, n_ent)            # si k > n, se usan todos los puntos
        D = self._distancias(X)           # (n_prueba, n_ent)

        # 1. Los k más cercanos (sin ordenar todo: argpartition es O(n)).
        if k < n_ent:
            idx = np.argpartition(D, k - 1, axis=1)[:, :k]
        else:
            idx = np.tile(np.arange(n_ent), (X.shape[0], 1))

        # 2. Ordenarlos por distancia (solo k elementos por fila): el orden
        #    se usa para el desempate.
        d_vec = np.take_along_axis(D, idx, axis=1)
        orden = np.argsort(d_vec, axis=1, kind="stable")
        idx = np.take_along_axis(idx, orden, axis=1)
        d_vec = np.take_along_axis(d_vec, orden, axis=1)

        # 3. Peso del voto de cada vecino.
        if self.ponderado:
            pesos = 1.0 / (d_vec + 1e-9)
        else:
            pesos = np.ones_like(d_vec)
        # Desempate: el vecino en la posición r aporta un extra de BONO/(r+1).
        pesos = pesos + _BONO_DESEMPATE / np.arange(1, k + 1)[None, :]

        # 4. Votación: suma de pesos por clase.
        etiquetas = self._y_idx[idx]                       # (n_prueba, k)
        votos = np.zeros((X.shape[0], len(self.classes_)))
        filas = np.repeat(np.arange(X.shape[0]), k)
        np.add.at(votos, (filas, etiquetas.ravel()), pesos.ravel())

        return self.classes_[np.argmax(votos, axis=1)]
