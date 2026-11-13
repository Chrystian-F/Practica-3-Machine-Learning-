# Naïve Bayes y parámetros (Rol 4)

## Cómo funciona

Naïve Bayes es un clasificador probabilístico basado en el teorema de Bayes. Para un punto $x = (x_1, \dots, x_d)$, la probabilidad de que pertenezca a la clase $c$ es:

$$P(c \mid x) = \frac{P(x \mid c)\, P(c)}{P(x)}$$

Como $P(x)$ es igual para todas las clases, basta comparar el numerador; el modelo asigna la clase con mayor probabilidad a posteriori. El supuesto "ingenuo" es la **independencia condicional**: dada la clase, los rasgos son independientes entre sí. Así, una distribución conjunta difícil de estimar se convierte en un producto de distribuciones de un solo rasgo:

$$P(x \mid c) = \prod_{j=1}^{d} P(x_j \mid c)$$

En la versión **gaussiana** (`GaussianNB`, vía `crear_naive_bayes()` en `modelos/bibliotecas.py`) cada $P(x_j \mid c)$ es una normal con media $\mu_{cj}$ y varianza $\sigma_{cj}^2$ estimadas con los puntos de entrenamiento de la clase $c$:

$$P(x_j \mid c) = \frac{1}{\sqrt{2\pi\sigma_{cj}^2}} \exp\!\left(-\frac{(x_j - \mu_{cj})^2}{2\sigma_{cj}^2}\right)$$

- **Entrenar:** para cada clase se calcula su prior $P(c) = n_c / n$ y la media y la varianza de cada rasgo ($2cd$ parámetros). No hay iteraciones ni semilla: el resultado siempre es el mismo.
- **Predecir:** se usan logaritmos para evitar productos de números muy pequeños: $\hat{y} = \arg\max_c \left[\log P(c) + \sum_j \log P(x_j \mid c)\right]$.
- **`var_smoothing`** (1e-9 por defecto) suma a todas las varianzas esa fracción de la varianza máxima entre los rasgos, para evitar divisiones entre cero y dar estabilidad numérica.

**Relación con el clasificador euclidiano.** Si las clases tienen el mismo prior y todos los rasgos comparten la misma varianza $\sigma^2$, el logaritmo de la probabilidad se reduce a $-\lVert x - \mu_c \rVert^2 / (2\sigma^2)$ más una constante, y la regla de decisión es asignar el punto al centroide más cercano en distancia euclidiana: es decir, Distancia Mínima. Naïve Bayes la generaliza al permitir una varianza distinta por clase y por rasgo, y al incorporar los priors; por eso su frontera puede ser curva y la de Distancia Mínima es lineal.

**Limitaciones.** El supuesto de independencia se viola cuando hay rasgos correlacionados o redundantes (el sintético tiene 2 redundantes), y el supuesto gaussiano no se cumple si una clase tiene varios grupos (el sintético tiene 2 por clase): una sola campana por clase y rasgo no puede describir una clase partida en dos.

**Escalado.** `GaussianNB` es casi invariante al escalado de los rasgos (cada rasgo tiene su propia media y varianza); el `StandardScaler` del `Pipeline` solo influye en cuánto pesa `var_smoothing` frente a las varianzas.

## Resultados con la evaluación común

Naïve Bayes con valores por defecto, evaluado con `evaluar_holdout` y `evaluar_kfold` (mismas particiones que los demás modelos). Precisión, recall y F1 con promedio macro; en K-Fold, promedio ± desviación estándar entre los 10 folds.

| Dataset | Validación | Accuracy | Precisión | Recall | F1 |
|---|---|---|---|---|---|
| Sintético | Hold-Out 70/30 | 0.5833 | 0.6069 | 0.5867 | 0.5800 |
| Sintético | K-Fold (k = 10) | 0.6083 ± 0.0611 | 0.6339 | 0.6109 | 0.6002 |
| Wine | Hold-Out 70/30 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Wine | K-Fold (k = 10) | 0.9778 ± 0.0369 | 0.9800 | 0.9794 | 0.9783 |

**Análisis.**

- **Wine: de los mejores.** En Hold-Out acierta las 54 muestras (empatado con el MLP) y en K-Fold 174 de 178: los 4 errores son 2 vinos de la clase 0 clasificados como clase 1 y 2 de la clase 1 clasificados como clase 2. Las clases de Wine están bien separadas y sus rasgos, una vez estandarizados, se parecen bastante a campanas por clase, que es justo lo que supone `GaussianNB`.
- **Sintético: el más bajo de los modelos programados.** Obtiene 0.5833 en Hold-Out (105 de 180) y 0.6083 en K-Fold (365 de 600), contra 0.68 a 0.71 de SVM y MLP. Es el dataset donde fallan sus supuestos: cada clase tiene 2 grupos (una sola gaussiana por clase y rasgo no los describe), hay 2 rasgos redundantes (no son independientes) y el 5 % de las etiquetas está cambiado.
- **La clase 1 es la más difícil.** Su sensibilidad es 0.39 en Hold-Out y 0.41 en K-Fold (contra 0.68 a 0.72 de las clases 0 y 2); en K-Fold, 95 de sus 205 muestras se clasifican como clase 0.
- **Una sola partición no alcanza.** En el sintético la accuracy por fold varía bastante (desviación de 0.06), así que la diferencia entre Hold-Out y K-Fold (0.58 contra 0.61) cabe dentro de esa variación.

## Ajuste de parámetros

`GaussianNB` casi no tiene hiperparámetros. Como prueba extra se evaluó `var_smoothing` ∈ {1e-9, 1e-7, 1e-5} con la evaluación común (script `ajuste_naive_bayes.py`; tabla completa en `resultados/ajuste_naive_bayes.csv`):

| Dataset | var_smoothing | Accuracy Hold-Out | Accuracy K-Fold (k = 10) |
|---|---|---|---|
| Sintético | 1e-9, 1e-7, 1e-5 | 0.5833 (los tres iguales) | 0.6083 ± 0.0611 (los tres iguales) |
| Wine | 1e-9, 1e-7, 1e-5 | 1.0000 (los tres iguales) | 0.9778 ± 0.0369 (los tres iguales) |

**¿Cambia algo? No.** Los tres valores dan exactamente el mismo resultado. Con los datos estandarizados las varianzas de los rasgos son cercanas a 1, así que sumarles 1e-5 o menos no modifica las predicciones. Se conserva el valor por defecto (1e-9). Mejorar Naïve Bayes en el sintético no depende de este parámetro, sino de que el modelo dejaría de ser tan restrictivo (por ejemplo, otro tipo de modelo).

## ¿Cómo encontrar los mejores parámetros para cada dataset?

Los parámetros óptimos dependen de los datos, así que se buscan por separado en cada dataset:

1. Definir el espacio de búsqueda de cada modelo (por ejemplo, k y la métrica en K-NN; kernel, C y gamma en SVM; capas, neuronas y activación en MLP).
2. **Búsqueda en malla (`GridSearchCV`):** prueba todas las combinaciones del espacio. Es exhaustiva y reproducible, pero su costo crece muy rápido con cada parámetro nuevo (la SVM de esta práctica son 52 combinaciones, 260 entrenamientos por dataset).
3. **Búsqueda aleatoria (`RandomizedSearchCV`):** evalúa solo `n_iter` combinaciones tomadas al azar (o de distribuciones, por ejemplo log-uniformes para C y gamma). Es más barata y conviene cuando el espacio es grande o tiene parámetros continuos.
4. **Validación cruzada solo sobre los datos de entrenamiento** (aquí `cv = 5` sobre el 70 %): cada combinación se puntúa con su accuracy promedio en los folds.
5. Elegir la combinación con mejor puntuación (`best_params_`), reentrenar con todo el entrenamiento y **evaluar una sola vez en el conjunto de prueba (30 %)**, que nunca participa en la decisión.
6. Poner el escalador dentro de un `Pipeline`, para que se ajuste solo con el entrenamiento de cada fold; por eso los parámetros llevan prefijo (`modelo__k`, `modelo__C`, …).

Escoger parámetros con el conjunto de prueba, o escalar todo el dataset antes de validar, filtra información (*data leakage*) y da resultados demasiado optimistas. Además, si varias combinaciones empatan (como en la SVM sobre Wine), conviene preferir la más simple, y si la diferencia es menor que la desviación entre folds, no se puede afirmar que la mejora sea real.

## Complejidad temporal

Notación: n = puntos de entrenamiento, d = rasgos, c = clases.

- **Entrenamiento: O(n·d).** Una sola pasada por los datos para calcular la media y la varianza de cada rasgo en cada clase.
- **Predicción: O(c·d) por punto.** Evalúa d gaussianas por cada una de las c clases; no depende de n.
- **Memoria: O(c·d)** (una media y una varianza por clase y rasgo).

**Lo que se observó** (`resultados/resultados.csv`): entrena en cerca de 1 ms y predice en cerca de 0.3 ms con ambos datasets, y esos tiempos casi no cambian entre Wine (n = 124, d = 13) y el sintético (n = 420, d = 6), coherente con un costo bajo y que no depende de n al predecir. A estos tamaños los tiempos son muy pequeños y variables, por lo que la comparación con la teoría corresponde a la prueba de escalabilidad (Rol 6).
