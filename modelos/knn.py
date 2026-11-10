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
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class KNN(BaseEstimator, ClassifierMixin):

    def __init__(self, k=5, metrica="euclidiana", ponderado=False):
        # REGLA de scikit-learn: en __init__ solo se guardan los parámetros,
        # con el mismo nombre que el argumento. Nada de cálculos aquí.
        self.k = k
        self.metrica = metrica        # "euclidiana" o "manhattan"
        self.ponderado = ponderado    # True -> voto pesado por 1/(distancia + 1e-9)

    def fit(self, X, y):
        """
        TODO (Rol 3):
          1. Guardar self.X_ = X, self.y_ = y, self.classes_ = np.unique(y)
          2. return self
        """
        raise NotImplementedError("KNN.fit aún no está implementado (Rol 3)")

    def predict(self, X):
        """
        TODO (Rol 3):
          1. Matriz de distancias (n_prueba, n_entrenamiento) según self.metrica,
             vectorizada.
          2. Índices de los k vecinos más cercanos (np.argpartition).
          3. Votación (mayoritaria o ponderada) y regla de desempate.
          4. return etiquetas predichas
        """
        raise NotImplementedError("KNN.predict aún no está implementado (Rol 3)")
