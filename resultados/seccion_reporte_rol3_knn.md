# K-NN y 1-NN (Rol 3)

## Cómo funciona

K-NN es un clasificador basado en memoria: la fase de entrenamiento solo guarda el conjunto de entrenamiento y no aprende ningún parámetro. Para clasificar un punto nuevo se calcula su distancia a todos los puntos guardados, se toman los k más cercanos y se asigna la clase que más veces aparece entre ellos. En esta práctica se programó desde cero con NumPy en una sola clase `KNN(k, metrica, ponderado)`; 1-NN es el caso `KNN(k=1)`.

- **Distancias.** Euclidiana (calculada de forma vectorizada con ‖a−b‖² = ‖a‖² + ‖b‖² − 2·a·b) o Manhattan (suma de diferencias absolutas, por bloques para limitar la memoria).
- **Vecinos.** Se usa `np.argpartition` para obtener los k más cercanos sin ordenar todas las distancias.
- **Voto.** Mayoritario, o ponderado (cada vecino pesa 1/(distancia + 1e-9), de modo que los vecinos cercanos influyen más).
- **Desempates.** Si dos clases empatan en votos, gana la clase cuyos vecinos están más cerca (el vecino más cercano vale un poco más que el segundo, y así sucesivamente). Con k = 1 no hay empates.
- **Escalado.** El modelo no escala los datos: el `StandardScaler` va dentro del `Pipeline` de `evaluacion.py`, ajustado solo con la parte de entrenamiento de cada partición. Esto importa especialmente en K-NN, porque las distancias dependen de la escala de cada rasgo (en Wine, por ejemplo, la prolina vale cientos y otros rasgos valen décimas).

Como verificación, se comparó la implementación con `KNeighborsClassifier` de scikit-learn (solo para comprobar, no se usa en el proyecto): en Wine y con k = 1 las predicciones coinciden al 100 %. Las únicas diferencias aparecen en la votación sin ponderar con k > 1 en el dataset sintético (94–98 % de coincidencia), y vienen de que scikit-learn rompe los empates por el número de clase, mientras que aquí se rompen por cercanía.

## Por qué 1-NN es sensible al ruido, y cómo k controla sesgo y varianza

Con k = 1 cada punto de prueba copia la etiqueta de un único vecino; si ese vecino tiene una etiqueta equivocada (el dataset sintético tiene 5 % de etiquetas cambiadas a propósito) o es un punto atípico, el error se propaga directamente. La frontera de decisión resultante es muy irregular: sesgo bajo y varianza alta (sobreajuste). Al aumentar k, el voto promedia más vecinos y los errores aislados se compensan: la frontera se suaviza y la varianza baja, a costa de más sesgo. Si k es demasiado grande, la predicción se parece a "la clase más común de la zona" y se pierde detalle (subajuste).

La gráfica de accuracy contra k (`resultados/graficas/knn_accuracy_vs_k.png`) lo muestra en el dataset sintético: la accuracy en validación cruzada sube de 0.650 con k = 1 hasta 0.710 con k = 11 (euclidiana, sin ponderar) y baja a 0.700 con k = 15.

## Ajuste de parámetros

Se probaron k ∈ {1, 3, 5, 7, 9, 11, 15}, distancia euclidiana y Manhattan, y voto ponderado sí/no (28 combinaciones) con `GridSearchCV(cv=5)` usando solo el 70 % de entrenamiento (partición estratificada con semilla 42). Después, la mejor combinación y la configuración por defecto (k = 5, euclidiana, sin ponderar) se evaluaron una sola vez en el 30 % de prueba.

| Dataset | Parámetros por defecto | Mejores parámetros | Accuracy CV (defecto → mejor) | Accuracy prueba (defecto → mejor) | Diferencia |
|---|---|---|---|---|---|
| Sintético | k=5, euclidiana, sin ponderar | k=11, euclidiana, sin ponderar | 0.6905 → 0.7095 | 0.6222 → 0.7056 | +0.0833 |
| Wine | k=5, euclidiana, sin ponderar | k=5, Manhattan, sin ponderar | 0.9433 → 0.9837 | 0.9444 → 0.9815 | +0.0370 |

**¿Mejoran los resultados?** Sí, en ambos datasets, tanto en validación cruzada como en el conjunto de prueba. Aun así, hay que interpretar las cifras con cuidado: el conjunto de prueba es pequeño (54 muestras en Wine, 180 en el sintético), por lo que la mejora de +0.037 en Wine equivale a 2 muestras más bien clasificadas. La mejora en validación cruzada es la más confiable y apunta en la misma dirección. En Wine, la mejora viene de usar la distancia Manhattan; en el sintético, de usar un k mayor (más vecinos promedian el ruido). El voto ponderado no aportó mejoras: en Wine da exactamente la misma accuracy que el voto sin ponderar y en el sintético queda igual o ligeramente por debajo.

Como referencia con la evaluación común (`main.py`): 1-NN obtiene 0.6278 (Hold-Out) y 0.6550 ± 0.0727 (K-Fold) en el sintético, y 0.9630 y 0.9552 ± 0.0417 en Wine; K-NN con k = 5 obtiene 0.6222 y 0.6950 ± 0.0719 en el sintético, y 0.9444 y 0.9722 ± 0.0448 en Wine.

## Complejidad temporal

- **Entrenamiento:** O(1) en tiempo, porque solo se guardan referencias a los datos; la memoria es O(n·d) (n = muestras de entrenamiento, d = rasgos).
- **Predicción (por punto):** O(n·d) para calcular las distancias a todo el entrenamiento, más O(n) para seleccionar los k más cercanos con `argpartition` y O(k log k) para ordenarlos. Domina O(n·d), así que el tiempo de predicción crece de forma lineal con n; por eso K-NN es rápido de "entrenar" pero lento de usar con conjuntos grandes.
