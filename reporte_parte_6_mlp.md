# Parte 6 — Perceptrón multicapa (MLP)

## Funcionamiento

Un perceptrón multicapa es una red neuronal formada por una capa de entrada, una o más capas ocultas y una capa de salida. Cada neurona combina sus entradas con pesos y un sesgo; después aplica una función de activación. En esta práctica se usa `MLPClassifier` de scikit-learn para clasificar las tres clases de cada dataset.

Durante el entrenamiento, la propagación hacia adelante calcula la salida de la red. La función de pérdida mide el error entre esa salida y la clase correcta. Backpropagation propaga el gradiente de ese error desde la salida hacia las capas anteriores y el optimizador Adam actualiza los pesos. El proceso se repite por épocas hasta cumplir el criterio de convergencia o llegar al máximo de iteraciones.

## Ajuste de hiperparámetros

Se comparó el MLP de referencia —una capa oculta de 100 neuronas, activación ReLU, `learning_rate_init=0.001` y `alpha=0.0001`— con una búsqueda exhaustiva en malla. Se probaron arquitecturas `(50,)`, `(100,)`, `(50, 50)` y `(100, 50)`; activaciones ReLU y tanh; tasas de aprendizaje iniciales 0.001 y 0.01; y regularización `alpha` de 0.0001 y 0.001. La búsqueda usa cinco folds y accuracy como criterio.

En cada dataset, el 30 % se reserva para la evaluación final. `GridSearchCV` recibe únicamente el 70 % de entrenamiento. El `StandardScaler` está dentro del pipeline, por lo que se ajusta por separado dentro de cada fold. La accuracy final se calcula una vez en el conjunto de prueba y se compara con la configuración de referencia.

La tabla completa queda en `resultados/ajuste_mlp.csv`; el detalle de las combinaciones probadas, en `resultados/ajuste_mlp_rejilla.csv`; y la curva de pérdida del mejor estimador de cada dataset, en `resultados/graficas/mlp_curva_perdida.png`.

## Complejidad temporal

Sea `n` el número de muestras, `W` el número total de pesos y sesgos, y `E` la cantidad de épocas ejecutadas. En una pasada por los datos, el cálculo hacia adelante y backpropagation cuesta aproximadamente `O(n·W)`. Por tanto, el entrenamiento cuesta `O(E·n·W)`. Para una observación nueva, la propagación hacia adelante cuesta `O(W)`. La memoria del modelo es `O(W)`, además de los datos y activaciones que necesita el entrenamiento.

Estas expresiones son aproximaciones: el tiempo también depende de la arquitectura, la activación, el criterio de convergencia, el hardware y el tamaño del lote. Al mantener una arquitectura fija y aumentar `n`, se espera que el entrenamiento crezca aproximadamente con `n`; la predicción de un lote fijo depende principalmente de `W`, no del número de muestras usado para entrenar.

## Resultados del ajuste MLP

La búsqueda probó 32 combinaciones por dataset, con cinco folds y usando solo el conjunto de entrenamiento. La tabla muestra los promedios de validación cruzada y la evaluación final en el 30 % reservado para prueba.

| Dataset | Mejores parámetros | Accuracy CV defecto | Accuracy CV mejor | Accuracy prueba defecto | Accuracy prueba mejor | Diferencia en prueba |
|---|---|---:|---:|---:|---:|---:|
| Sintético | `hidden_layer_sizes=(50,)`, ReLU, `learning_rate_init=0.001`, `alpha=0.001` | 0.7024 | 0.7095 | 0.6833 | 0.6889 | +0.0056 |
| Wine | `hidden_layer_sizes=(50, 50)`, ReLU, `learning_rate_init=0.01`, `alpha=0.0001` | 0.9840 | 0.9920 | 1.0000 | 0.9815 | -0.0185 |

En el dataset sintético, el mejor ajuste sube la accuracy de prueba en 0.0056 (una muestra adicional de las 180 de prueba). En Wine, la búsqueda mejora el promedio de validación cruzada, pero en la partición de prueba pierde una muestra: el modelo por defecto acertó las 54 y el ajustado acertó 53. Por eso, la mejora en CV no garantiza una mejora en cada partición de prueba.

La curva de pérdida del mejor modelo desciende de 1.2043 a 0.5475 en 559 iteraciones para el dataset sintético, y de 1.2783 a 0.0000816 en 32 iteraciones para Wine. Los valores por iteración están en `resultados/ajuste_mlp_curvas_perdida.csv`; las configuraciones probadas están en `resultados/ajuste_mlp_rejilla.csv`.

## Comparación de tiempos

`escalabilidad.py` crea datasets sintéticos con 200, 400, 800, 1600 y 3200 observaciones. Mide por separado el entrenamiento y la predicción, promedia tres corridas por tamaño y mantiene fijo el lote de predicción en 100 observaciones. Los tiempos del MLP y de los otros clasificadores se guardan en `resultados/escalabilidad_tiempos.csv`; la gráfica logarítmica se guarda en `resultados/graficas/escalabilidad_tiempos.png`.

La comparación empírica debe interpretarse como una comprobación de tendencias, no como una igualdad exacta entre segundos y Big-O. Para el MLP se espera que el entrenamiento aumente con `n`, mientras que la predicción por una cantidad fija de observaciones permanezca más estable. Las diferencias entre corridas y el costo de inicialización pueden ser notables cuando los datasets son pequeños.

## Estado de resultados

El ajuste MLP ya se ejecutó y los CSV y PNG descritos arriba están guardados en `resultados/`. La escalabilidad contiene 30 filas: cinco tamaños para cada uno de los seis clasificadores. Se midieron cinco; `DistanciaMinima` queda con estado `pendiente` porque su `fit` todavía lanza `NotImplementedError`. Cuando el rol 2 integre esa clase, se debe volver a ejecutar `escalabilidad.py` para completar sus tiempos.

La gráfica usa escala logarítmica. Para el MLP, el entrenamiento tarda 1.1412 s con 200 muestras y 6.5608 s con 3200; la predicción de un lote fijo de 100 muestras se mantiene entre 0.0009 y 0.0013 s. El entrenamiento crece con `n`, aunque no de forma perfectamente lineal: el número de iteraciones hasta converger cambia entre tamaños. La predicción se mantiene estable, como anticipa `O(W)` por muestra cuando la arquitectura no cambia.

La medición incluye el `StandardScaler` del pipeline. Por eso el entrenamiento medido de K-NN también contiene el costo de escalar los datos (`O(n·d)`), aunque el `fit` del clasificador K-NN por sí mismo solo memoriza las observaciones. En la gráfica, K-NN y 1-NN muestran el crecimiento principalmente en la predicción; SVM aumenta más en entrenamiento; Naïve Bayes conserva tiempos bajos. Las cifras exactas de los cinco modelos medidos están en `resultados/escalabilidad_tiempos.csv`.

## Ejecución

```bash
pip install -r requirements.txt
python -c "from ajuste_parametros import ajustar_mlp; ajustar_mlp()"
python escalabilidad.py --repeticiones 3
```
