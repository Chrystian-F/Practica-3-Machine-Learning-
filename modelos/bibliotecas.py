"""
modelos/bibliotecas.py
----------------------
Modelos implementados CON BIBLIOTECA (scikit-learn).

Cada función regresa un modelo nuevo SIN entrenar, con los parámetros por
defecto que se usan en la comparación principal. Los parámetros ajustados
(mejores encontrados con GridSearchCV) se reportan aparte en
ajuste_parametros.py; no cambien estos valores por defecto.

RESPONSABLES:
    crear_naive_bayes -> Rol 4
    crear_svm         -> Rol 5
    crear_mlp         -> Rol 6
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

SEMILLA = 42


def crear_naive_bayes():
    """Naïve Bayes gaussiano con valores por defecto (var_smoothing=1e-9)."""
    return GaussianNB()


def crear_svm():
    """SVM con valores por defecto: kernel='rbf', C=1.0, gamma='scale'."""
    return SVC(random_state=SEMILLA)


def crear_mlp():
    """
    Perceptrón multicapa con valores por defecto: una capa oculta de 100
    neuronas, activación relu, optimizador adam, learning_rate_init=0.001.

    Único cambio: max_iter=1000 (el default de 200 no alcanza a converger y
    scikit-learn lanza un ConvergenceWarning).
    """
    return MLPClassifier(max_iter=1000, random_state=SEMILLA)
