"""
modelos/distancia_minima.py
---------------------------
RESPONSABLE: Rol 2.

Clasificador de Distancia Mínima (implementación DESDE CERO, solo NumPy).
Prohibido usar NearestCentroid u otros clasificadores de scikit-learn.

Idea: en fit se calcula el centroide (promedio) de cada clase; en predict
cada punto recibe la clase del centroide más cercano (distancia euclidiana).

Se hereda de BaseEstimator y ClassifierMixin para que funcione dentro de
Pipeline, cross_val_score y GridSearchCV igual que un modelo de scikit-learn.
NO escalen los datos aquí: el escalado ya lo hace el Pipeline de evaluacion.py.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class DistanciaMinima(BaseEstimator, ClassifierMixin):

    def __init__(self):
        # Si agregan parámetros (p. ej. metrica="euclidiana"), deben guardarse
        # aquí con el MISMO nombre: self.metrica = metrica. Nada más en __init__.
        pass

    def fit(self, X, y):
        """
        TODO (Rol 2):
          1. self.classes_ = np.unique(y)
          2. self.centroides_ = matriz (n_clases, n_rasgos) con el promedio
             de los puntos de cada clase.
          3. return self
        """
        raise NotImplementedError("DistanciaMinima.fit aún no está implementado (Rol 2)")

    def predict(self, X):
        """
        TODO (Rol 2):
          1. Calcular la distancia de cada punto de X a cada centroide,
             vectorizado (sin ciclo por punto). Resultado: (n_puntos, n_clases).
          2. Para cada punto, índice del centroide más cercano (np.argmin).
          3. return self.classes_[indices]
        """
        raise NotImplementedError("DistanciaMinima.predict aún no está implementado (Rol 2)")
