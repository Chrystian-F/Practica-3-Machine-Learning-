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
    """
    Máquina de vectores de soporte (SVM) con los valores por defecto de SVC,
    escritos de forma explícita para que se vean en el código:

      - kernel="rbf"   frontera no lineal: K(x, z) = exp(-gamma * ||x - z||²).
      - C=1.0          costo de violar el margen (variables de holgura).
      - gamma="scale"  1 / (n_rasgos * varianza de X). Como el Pipeline de
                       evaluacion.py ya estandariza los datos (varianza 1),
                       equivale a 1 / n_rasgos.

    Con más de dos clases, SVC entrena un clasificador por cada par de clases
    (uno contra uno) y decide por votación. random_state no influye en el
    resultado (SVC solo lo usa para estimar probabilidades, que aquí no se
    piden); se deja por consistencia con los demás modelos.
    """
    return SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEMILLA)


def crear_mlp():
    """
    Perceptrón multicapa con valores por defecto: una capa oculta de 100
    neuronas, activación relu, optimizador adam, learning_rate_init=0.001.

    Único cambio: max_iter=1000 (el default de 200 no alcanza a converger y
    scikit-learn lanza un ConvergenceWarning).
    """
    return MLPClassifier(max_iter=1000, random_state=SEMILLA)
