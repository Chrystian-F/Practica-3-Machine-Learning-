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

## Comparación de tiempos

`escalabilidad.py` crea datasets sintéticos con 200, 400, 800, 1600 y 3200 observaciones. Mide por separado el entrenamiento y la predicción, promedia tres corridas por tamaño y mantiene fijo el lote de predicción en 100 observaciones. Los tiempos del MLP y de los otros clasificadores se guardan en `resultados/escalabilidad_tiempos.csv`; la gráfica logarítmica se guarda en `resultados/graficas/escalabilidad_tiempos.png`.

La comparación empírica debe interpretarse como una comprobación de tendencias, no como una igualdad exacta entre segundos y Big-O. Para el MLP se espera que el entrenamiento aumente con `n`, mientras que la predicción por una cantidad fija de observaciones permanezca más estable. Las diferencias entre corridas y el costo de inicialización pueden ser notables cuando los datasets son pequeños.

## Estado de resultados

Los archivos con valores numéricos se generan al ejecutar los scripts en un entorno que tenga instaladas las dependencias de `requirements.txt`. En la copia revisada, `DistanciaMinima` todavía lanza `NotImplementedError`; por eso la tabla de escalabilidad lo registra como pendiente y mide el resto de modelos. Cuando el rol 2 integre esa clase, se debe volver a ejecutar `escalabilidad.py` para completar la comparación de los seis clasificadores.

## Ejecución

```bash
pip install -r requirements.txt
python -c "from ajuste_parametros import ajustar_mlp; ajustar_mlp()"
python escalabilidad.py --repeticiones 3
```
