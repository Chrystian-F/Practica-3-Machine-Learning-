
"""
modelos/distancia_minima.py
---------------------------
RESPONSABLE: Rol 2.

Clasificador de Distancia Mínima (implementación DESDE CERO, solo NumPy).
Prohibido usar NearestCentroid u otros clasificadores de scikit-learn.

Idea:
En fit se calcula el centroide (promedio) de cada clase.
En predict cada punto recibe la clase del centroide más cercano
mediante distancia euclidiana.

Se hereda de BaseEstimator y ClassifierMixin para que funcione
dentro de Pipeline, cross_val_score y GridSearchCV.

NO escalar los datos aquí: el escalado ya lo realiza
el Pipeline de evaluacion.py.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class DistanciaMinima(BaseEstimator, ClassifierMixin):

    def __init__(self):
        pass

    def fit(self, X, y):
        """
        Entrena el clasificador calculando el centroide de cada clase.
        """

        X = np.asarray(X)
        y = np.asarray(y)

        # Verificar dimensiones
        if X.ndim != 2:
            raise ValueError("X debe ser una matriz bidimensional.")

        if y.ndim != 1:
            raise ValueError("y debe ser un vector unidimensional.")

        if len(X) != len(y):
            raise ValueError("X e y deben tener el mismo número de muestras.")

        if len(X) == 0:
            raise ValueError("No se puede entrenar con un conjunto vacío.")

        # 1. Obtener las clases existentes
        self.classes_ = np.unique(y)

        # 2. Calcular el centroide de cada clase
        self.centroides_ = np.array([
            X[y == clase].mean(axis=0)
            for clase in self.classes_
        ])

        # 3. Devolver el clasificador entrenado
        return self

    def predict(self, X):
        """
        Predice la clase del centroide más cercano.
        """

        # Comprobar que el modelo esté entrenado
        if not hasattr(self, "centroides_"):
            raise ValueError("El clasificador debe entrenarse antes de predecir.")

        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError("X debe ser una matriz bidimensional.")

        if X.shape[1] != self.centroides_.shape[1]:
            raise ValueError("X tiene un número de características diferente al entrenamiento.")

        # 1. Calcular las distancias euclidianas de todos
        # los puntos a todos los centroides.
        diferencias = X[:, np.newaxis, :] - self.centroides_[np.newaxis, :, :]

        distancias = np.sqrt(np.sum(diferencias ** 2, axis=2))

        # 2. Obtener el índice del centroide más cercano
        indices = np.argmin(distancias, axis=1)

        # 3. Devolver las clases correspondientes
        return self.classes_[indices]
