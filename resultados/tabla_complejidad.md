# Complejidad temporal de los 6 clasificadores (tabla consolidada, Rol 5)

Notación: **n** = puntos de entrenamiento, **d** = rasgos, **c** = clases, **k** = vecinos, **s** = vectores de soporte (s ≤ n), **W** = total de pesos del MLP, **E** = épocas de entrenamiento. El costo de predicción es por cada punto nuevo.

| Modelo | Entrenamiento | Predicción (por punto) | Memoria del modelo | Comentario |
|---|---|---|---|---|
| Distancia Mínima | O(n·d) | O(c·d) | O(c·d) | Una pasada por los datos para promediar cada clase. Al predecir solo compara contra c centroides, sin importar n. Es el más barato de los seis; a cambio, su frontera es lineal. |
| 1-NN | O(1) | O(n·d) | O(n·d) | No aprende nada: solo guarda los datos ("aprendizaje perezoso"). Todo el costo se paga al predecir, que exige la distancia a los n puntos. |
| K-NN | O(1) | O(n·d) | O(n·d) | Igual que 1-NN, más O(n) para escoger los k más cercanos (`np.argpartition`) y O(k log k) para ordenarlos. Domina O(n·d), así que k casi no cambia el costo. |
| Naïve Bayes (gaussiano) | O(n·d) | O(c·d) | O(c·d) | Una pasada para calcular la media y la varianza de cada rasgo en cada clase (2·c·d parámetros). Al predecir evalúa d gaussianas por clase. Mismo orden que Distancia Mínima. |
| SVM (`SVC`) | Entre O(n²·d) y O(n³) | O(s·d) | O(s·d); al entrenar, hasta O(n²) de caché del kernel | Resuelve un problema cuadrático con n variables; el número de iteraciones depende de C y del kernel. Es el que más se encarece al crecer n. Al predecir solo usa los vectores de soporte, pero con datos ruidosos s se acerca a n (67 % de los puntos en el sintético, 46 % en Wine). |
| MLP | O(E·n·W) | O(W) | O(W) | Cada época recorre los n puntos y por cada uno hace una pasada hacia adelante y otra hacia atrás (backpropagation), ambas O(W). Es lineal en n, pero con una constante grande: E puede ser de cientos. Predecir no depende de n. |

## Cómo leer la tabla

- **Entrenamiento, por orden de crecimiento con n.** K-NN no hace nada; Distancia Mínima y Naïve Bayes dan una sola pasada por los datos; el MLP da E pasadas; la SVM crece entre cuadrática y cúbicamente. Si n se duplica, el costo de Distancia Mínima, Naïve Bayes y MLP se duplica, y el de la SVM se multiplica por un factor de 4 a 8.
- **La notación O dice cómo crece el costo, no cuánto tarda.** Con los tamaños de esta práctica (n de cientos) el MLP puede tardar más que la SVM en entrenar, porque su constante E·W es grande aunque crezca linealmente.
- **Predicción.** Distancia Mínima, Naïve Bayes y MLP tienen un costo fijo que no depende de cuántos datos se usaron para entrenar. K-NN se encarece en proporción a n, y la SVM en proporción a sus vectores de soporte.

## Tamaño de cada término en esta práctica

Con la partición Hold-Out (entrenamiento con el 70 %), el término que domina la predicción de un punto vale:

| Término | Modelos | Sintético (n = 420, d = 6, c = 3) | Wine (n = 124, d = 13, c = 3) |
|---|---|---|---|
| c·d | Distancia Mínima, Naïve Bayes | 18 | 39 |
| W | MLP por defecto (una capa oculta de 100 neuronas): W = 100·(d + c) | 900 | 1 600 |
| s·d | SVM por defecto (s = 282 en el sintético, s = 57 en Wine) | 1 692 | 741 |
| n·d | 1-NN, K-NN | 2 520 | 1 612 |

Son órdenes de magnitud del número de operaciones, no tiempos: las constantes de cada implementación (NumPy vectorizado, libsvm en C) cambian los tiempos reales. La comparación con tiempos medidos corresponde a la prueba de escalabilidad (Rol 6).
