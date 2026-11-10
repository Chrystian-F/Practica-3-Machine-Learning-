# Práctica 3: Machine Learning — Comparativa de clasificadores

Inteligencia Artificial · ESCOM-IPN

Comparativa de 6 clasificadores en 2 datasets con 2 métodos de validación.

| Modelo | Implementación | Archivo |
| --- | --- | --- |
| Distancia Mínima | Desde cero (NumPy) | `modelos/distancia_minima.py` |
| 1-NN | Desde cero (NumPy) | `modelos/knn.py` (`KNN(k=1)`) |
| K-NN | Desde cero (NumPy) | `modelos/knn.py` |
| Naïve Bayes | scikit-learn | `modelos/bibliotecas.py` |
| SVM | scikit-learn | `modelos/bibliotecas.py` |
| MLP | scikit-learn | `modelos/bibliotecas.py` |

- **Datasets:** sintético con ruido estructurado (`make_classification`) y Wine (13 rasgos, 3 clases).
- **Validación:** Hold-Out 70/30 estratificado y K-Fold estratificado con k = 10.
- **Métricas:** accuracy, precisión, recall y F1 (macro), matriz de confusión, sensibilidad y especificidad por clase, tiempos de entrenamiento y predicción.

## Cómo correrlo

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Los resultados quedan en `resultados/resultados.csv` y `resultados/matrices_confusion.json`.

## Estructura

```
practica3-ml/
├── datasets.py            carga y generación de datasets (NO modificar)
├── evaluacion.py          Hold-Out, K-Fold y métricas (NO modificar)
├── main.py                corre los 24 experimentos
├── ajuste_parametros.py   búsqueda de hiperparámetros de K-NN, SVM y MLP
├── modelos/
│   ├── distancia_minima.py
│   ├── knn.py
│   └── bibliotecas.py
├── resultados/            CSV, JSON y gráficas generadas
└── conversaciones_ia/     conversaciones con herramientas de IA (se entregan)
```

## Reglas para el equipo

1. Cada quien trabaja en su rama (`git checkout -b modelo-knn`) y abre un pull request a `main`.
2. Nadie modifica `datasets.py` ni `evaluacion.py`; si algo falla ahí, avisen al integrador.
3. Todo modelo tiene `fit(X, y)` y `predict(X)`. Los hechos desde cero heredan de `BaseEstimator` y `ClassifierMixin`.
4. Los modelos NO escalan los datos: el `StandardScaler` ya va dentro del `Pipeline` de `evaluacion.py`.
5. Antes de abrir su pull request, comprueben que `python main.py` corre sin errores.
6. Guarden su conversación con IA en `conversaciones_ia/<nombre>_<rol>.pdf` (o `.md`).
