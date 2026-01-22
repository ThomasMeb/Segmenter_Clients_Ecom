"""
Module d'évaluation des modèles de clustering.

Ce module fournit des métriques pour évaluer la qualité
des segmentations.
"""

from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score,
)


def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Évalue la qualité d'un clustering avec plusieurs métriques.

    Parameters
    ----------
    X : np.ndarray
        Features utilisées pour le clustering (scalées).
    labels : np.ndarray
        Labels des clusters.

    Returns
    -------
    dict
        Dictionnaire contenant :
        - silhouette : Silhouette Score [-1, 1], plus élevé = meilleur
        - calinski_harabasz : Calinski-Harabasz Index, plus élevé = meilleur
        - davies_bouldin : Davies-Bouldin Index, plus bas = meilleur
        - n_clusters : Nombre de clusters

    Examples
    --------
    >>> metrics = evaluate_clustering(X_scaled, labels)
    >>> print(f"Silhouette: {metrics['silhouette']:.3f}")
    """
    n_clusters = len(np.unique(labels))

    return {
        "silhouette": silhouette_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels),
        "n_clusters": n_clusters,
    }


def calculate_silhouette_score(
    X: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Calcule le Silhouette Score.

    Le Silhouette Score mesure à quel point les échantillons sont
    similaires à leur propre cluster par rapport aux autres clusters.

    Parameters
    ----------
    X : np.ndarray
        Features (scalées de préférence).
    labels : np.ndarray
        Labels des clusters.

    Returns
    -------
    float
        Score entre -1 et 1. Plus proche de 1 = meilleur clustering.
    """
    return silhouette_score(X, labels)


def calculate_calinski_harabasz_score(
    X: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Calcule le Calinski-Harabasz Index.

    Aussi appelé Variance Ratio Criterion, il mesure le ratio entre
    la dispersion inter-cluster et intra-cluster.

    Parameters
    ----------
    X : np.ndarray
        Features.
    labels : np.ndarray
        Labels des clusters.

    Returns
    -------
    float
        Score positif. Plus élevé = meilleur clustering.
    """
    return calinski_harabasz_score(X, labels)


def get_silhouette_per_sample(
    X: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    """
    Calcule le Silhouette Score pour chaque échantillon.

    Utile pour identifier les échantillons mal classés.

    Parameters
    ----------
    X : np.ndarray
        Features.
    labels : np.ndarray
        Labels des clusters.

    Returns
    -------
    pd.DataFrame
        DataFrame avec les colonnes 'cluster' et 'silhouette'.
    """
    sample_scores = silhouette_samples(X, labels)

    return pd.DataFrame({
        "cluster": labels,
        "silhouette": sample_scores,
    })


def compare_models(
    X: np.ndarray,
    models_labels: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Compare plusieurs modèles de clustering.

    Parameters
    ----------
    X : np.ndarray
        Features utilisées pour tous les modèles.
    models_labels : dict
        Dictionnaire {nom_modèle: labels}.

    Returns
    -------
    pd.DataFrame
        Comparaison des métriques pour chaque modèle.

    Examples
    --------
    >>> labels_dict = {
    ...     "KMeans k=3": kmeans3.labels_,
    ...     "KMeans k=4": kmeans4.labels_,
    ... }
    >>> comparison = compare_models(X_scaled, labels_dict)
    """
    results = []

    for model_name, labels in models_labels.items():
        metrics = evaluate_clustering(X, labels)
        metrics["model"] = model_name
        results.append(metrics)

    df = pd.DataFrame(results)
    df = df.set_index("model")

    return df.round(4)
