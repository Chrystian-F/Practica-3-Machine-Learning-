"""
evaluacion.py
-------------
Funciones comunes de validación: Hold-Out 70/30 y K-Fold Cross-Validation.

Cualquier modelo que tenga fit(X, y) y predict(X) se puede evaluar aquí:
los hechos desde cero (Distancia Mínima, KNN) y los de scikit-learn
(Naïve Bayes, SVM, MLP).

Puntos clave:
  1. El escalado (StandardScaler) va DENTRO de un Pipeline. Así el escalador
     aprende la media y desviación SOLO con la parte de entrenamiento de cada
     partición. Si escaláramos todo el dataset antes de dividir, la parte de
     prueba "se filtraría" al entrenamiento (data leakage) y los resultados
     saldrían mejores de lo real.
  2. Las particiones usan semilla fija y son estratificadas: todos los modelos
     se prueban con exactamente los mismos datos y cada partición conserva la
     proporción de clases.
  3. Se usa clone(modelo) en cada partición para empezar siempre con un
     modelo sin entrenar.
"""

import time

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEMILLA = 42

# Métricas escalares que se promedian en K-Fold.
METRICAS = ["accuracy", "precision", "recall", "f1"]


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------
def crear_pipeline(modelo):
    """Escalador + copia limpia del modelo. Se usa también en el ajuste de
    parámetros (GridSearchCV): ahí los nombres van como 'modelo__C'."""
    return Pipeline([
        ("escalador", StandardScaler()),
        ("modelo", clone(modelo)),
    ])


def calcular_metricas(y_real, y_pred, clases):
    """
    Accuracy, precisión, recall y F1 con promedio macro (todas las clases
    pesan igual), matriz de confusión, y sensibilidad / especificidad por
    clase (uno contra el resto).
    """
    mc = confusion_matrix(y_real, y_pred, labels=clases)

    # Por clase i:  VP = mc[i,i]   FN = resto de la fila   FP = resto de la columna
    vp = np.diag(mc)
    fn = mc.sum(axis=1) - vp
    fp = mc.sum(axis=0) - vp
    vn = mc.sum() - (vp + fn + fp)
    with np.errstate(divide="ignore", invalid="ignore"):
        sensibilidad = np.where(vp + fn > 0, vp / (vp + fn), 0.0)
        especificidad = np.where(vn + fp > 0, vn / (vn + fp), 0.0)

    return {
        "accuracy": accuracy_score(y_real, y_pred),
        "precision": precision_score(y_real, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_real, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_real, y_pred, average="macro", zero_division=0),
        "matriz_confusion": mc,
        "sensibilidad_por_clase": sensibilidad,
        "especificidad_por_clase": especificidad,
    }


def entrenar_y_probar(modelo, X_ent, y_ent, X_prueba, y_prueba, clases):
    """Entrena una copia del modelo, predice y mide tiempos por separado."""
    pipeline = crear_pipeline(modelo)

    inicio = time.perf_counter()
    pipeline.fit(X_ent, y_ent)
    t_entrenamiento = time.perf_counter() - inicio

    inicio = time.perf_counter()
    y_pred = pipeline.predict(X_prueba)
    t_prediccion = time.perf_counter() - inicio

    resultado = calcular_metricas(y_prueba, y_pred, clases)
    resultado["tiempo_entrenamiento_s"] = t_entrenamiento
    resultado["tiempo_prediccion_s"] = t_prediccion
    return resultado


# ---------------------------------------------------------------------------
# Métodos de validación
# ---------------------------------------------------------------------------
def evaluar_holdout(modelo, X, y, proporcion_prueba=0.30, random_state=SEMILLA):
    """
    Hold-Out 70/30 estratificado.

    Regresa un diccionario con accuracy, precision, recall, f1 (macro),
    matriz_confusion, sensibilidad_por_clase, especificidad_por_clase,
    tiempo_entrenamiento_s y tiempo_prediccion_s.
    """
    clases = np.unique(y)
    X_ent, X_prueba, y_ent, y_prueba = train_test_split(
        X, y,
        test_size=proporcion_prueba,
        stratify=y,
        random_state=random_state,
    )
    return entrenar_y_probar(modelo, X_ent, y_ent, X_prueba, y_prueba, clases)


def evaluar_kfold(modelo, X, y, k=10, random_state=SEMILLA):
    """
    K-Fold Cross-Validation estratificado (k=10 por defecto).

    Regresa un diccionario con:
      - <metrica> y <metrica>_std  para accuracy, precision, recall, f1
        (promedio y desviación estándar entre los k folds)
      - matriz_confusion: suma de las matrices de los k folds (cada muestra
        aparece exactamente una vez como prueba)
      - sensibilidad_por_clase, especificidad_por_clase: calculadas sobre
        esa matriz sumada
      - tiempo_entrenamiento_s, tiempo_prediccion_s: promedio por fold
      - accuracy_por_fold: lista con el accuracy de cada fold
    """
    clases = np.unique(y)
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

    por_fold = []
    mc_total = np.zeros((len(clases), len(clases)), dtype=int)

    for idx_ent, idx_prueba in kf.split(X, y):
        r = entrenar_y_probar(modelo, X[idx_ent], y[idx_ent],
                              X[idx_prueba], y[idx_prueba], clases)
        por_fold.append(r)
        mc_total += r["matriz_confusion"]

    resultado = {}
    for m in METRICAS:
        valores = [r[m] for r in por_fold]
        resultado[m] = float(np.mean(valores))
        resultado[f"{m}_std"] = float(np.std(valores))

    # Sensibilidad / especificidad sobre la matriz acumulada
    vp = np.diag(mc_total)
    fn = mc_total.sum(axis=1) - vp
    fp = mc_total.sum(axis=0) - vp
    vn = mc_total.sum() - (vp + fn + fp)
    resultado["matriz_confusion"] = mc_total
    resultado["sensibilidad_por_clase"] = vp / np.maximum(vp + fn, 1)
    resultado["especificidad_por_clase"] = vn / np.maximum(vn + fp, 1)

    resultado["tiempo_entrenamiento_s"] = float(np.mean([r["tiempo_entrenamiento_s"] for r in por_fold]))
    resultado["tiempo_prediccion_s"] = float(np.mean([r["tiempo_prediccion_s"] for r in por_fold]))
    resultado["accuracy_por_fold"] = [float(r["accuracy"]) for r in por_fold]
    return resultado


if __name__ == "__main__":
    # Prueba rápida: python evaluacion.py
    from sklearn.naive_bayes import GaussianNB
    from datasets import cargar_benchmark

    X, y = cargar_benchmark()
    ho = evaluar_holdout(GaussianNB(), X, y)
    kf = evaluar_kfold(GaussianNB(), X, y, k=10)
    print(f"Hold-Out  accuracy = {ho['accuracy']:.4f}")
    print(f"K-Fold 10 accuracy = {kf['accuracy']:.4f} ± {kf['accuracy_std']:.4f}")
    print("Matriz de confusión (Hold-Out):\n", ho["matriz_confusion"])
