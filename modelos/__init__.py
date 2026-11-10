"""Paquete con los 6 clasificadores de la Práctica 3."""

from .bibliotecas import crear_mlp, crear_naive_bayes, crear_svm
from .distancia_minima import DistanciaMinima
from .knn import KNN

__all__ = ["DistanciaMinima", "KNN", "crear_naive_bayes", "crear_svm", "crear_mlp"]
