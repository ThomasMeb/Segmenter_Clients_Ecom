"""
Evaluation metrics for clustering models.

This module provides functions to evaluate clustering quality using:
- Silhouette Score
- Calinski-Harabasz Score
- Davies-Bouldin Score
- Adjusted Rand Index (for stability analysis)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score
)
from typing import Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClusteringEvaluator:
    """
    Evaluates clustering model performance.

    Attributes:
        X: Feature matrix.
        labels: Cluster labels.
    """

    def __init__(self, X: np.ndarray, labels: np.ndarray):
        """
        Initialize the evaluator.

        Parameters:
            X (np.ndarray): Feature matrix (scaled).
            labels (np.ndarray): Cluster labels.
        """
        self.X = X
        self.labels = labels

    def compute_all_metrics(self) -> Dict[str, float]:
        """
        Compute all clustering evaluation metrics.

        Returns:
            Dict[str, float]: Dictionary of metric names and values.
        """
        # Remove noise points (label = -1) for DBSCAN
        mask = self.labels != -1
        X_filtered = self.X[mask]
        labels_filtered = self.labels[mask]

        n_clusters = len(np.unique(labels_filtered))

        if n_clusters < 2:
            logger.warning("Less than 2 clusters found. Metrics may not be meaningful.")
            return {
                'n_clusters': n_clusters,
                'n_noise': np.sum(self.labels == -1),
                'silhouette_score': None,
                'calinski_harabasz_score': None,
                'davies_bouldin_score': None
            }

        metrics = {
            'n_clusters': n_clusters,
            'n_noise': np.sum(self.labels == -1),
            'silhouette_score': silhouette_score(X_filtered, labels_filtered),
            'calinski_harabasz_score': calinski_harabasz_score(X_filtered, labels_filtered),
            'davies_bouldin_score': davies_bouldin_score(X_filtered, labels_filtered)
        }

        logger.info("\n=== Clustering Evaluation Metrics ===")
        logger.info(f"Number of clusters: {metrics['n_clusters']}")
        logger.info(f"Noise points (DBSCAN): {metrics['n_noise']}")
        logger.info(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
        logger.info(f"Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.2f}")
        logger.info(f"Davies-Bouldin Score: {metrics['davies_bouldin_score']:.4f}")
        logger.info("=====================================\n")

        return metrics

    def compute_silhouette_samples(self) -> np.ndarray:
        """
        Compute silhouette score for each sample.

        Returns:
            np.ndarray: Silhouette scores per sample.
        """
        mask = self.labels != -1
        X_filtered = self.X[mask]
        labels_filtered = self.labels[mask]

        if len(np.unique(labels_filtered)) < 2:
            logger.warning("Cannot compute sample silhouette scores with < 2 clusters")
            return None

        sample_scores = silhouette_samples(X_filtered, labels_filtered)
        return sample_scores


def evaluate_clustering(
    data: pd.DataFrame,
    labels: np.ndarray,
    feature_columns: list
) -> Dict[str, float]:
    """
    Evaluate clustering results.

    Parameters:
        data (pd.DataFrame): Customer data.
        labels (np.ndarray): Cluster labels.
        feature_columns (list): Feature columns used for clustering.

    Returns:
        Dict[str, float]: Evaluation metrics.
    """
    from sklearn.preprocessing import StandardScaler

    # Scale features
    scaler = StandardScaler()
    X = data[feature_columns].values
    X_scaled = scaler.fit_transform(X)

    # Compute metrics
    evaluator = ClusteringEvaluator(X_scaled, labels)
    metrics = evaluator.compute_all_metrics()

    return metrics


def compare_clusterings(
    labels_1: np.ndarray,
    labels_2: np.ndarray
) -> float:
    """
    Compare two clustering results using Adjusted Rand Index.

    Parameters:
        labels_1 (np.ndarray): First set of cluster labels.
        labels_2 (np.ndarray): Second set of cluster labels.

    Returns:
        float: Adjusted Rand Index (0 to 1, higher is more similar).
    """
    ari = adjusted_rand_score(labels_1, labels_2)
    logger.info(f"Adjusted Rand Index: {ari:.4f}")
    return ari


def temporal_stability_analysis(
    data: pd.DataFrame,
    model,
    feature_columns: list,
    date_column: str,
    n_weeks: int = 52
) -> pd.DataFrame:
    """
    Analyze clustering stability over time.

    Parameters:
        data (pd.DataFrame): Time-series customer data.
        model: Trained clustering model.
        feature_columns (list): Features for clustering.
        date_column (str): Column with date information.
        n_weeks (int): Number of weeks to analyze.

    Returns:
        pd.DataFrame: Stability metrics per week.
    """
    logger.info(f"Analyzing temporal stability over {n_weeks} weeks...")

    # Ensure date column is datetime
    df = data.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    # Get base clustering (week n_weeks)
    base_date = df[date_column].max() - pd.Timedelta(weeks=n_weeks)
    base_data = df[df[date_column] <= base_date]

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_base = scaler.fit_transform(base_data[feature_columns].values)
    base_labels = model.predict(X_base) if hasattr(model, 'predict') else model.fit_predict(X_base)

    # Compare with subsequent weeks
    results = []
    for week in range(1, n_weeks + 1):
        week_date = base_date + pd.Timedelta(weeks=week)
        week_data = df[df[date_column] <= week_date]

        if len(week_data) == 0:
            continue

        X_week = scaler.transform(week_data[feature_columns].values)
        week_labels = model.predict(X_week) if hasattr(model, 'predict') else None

        if week_labels is not None:
            # Compare only overlapping customers
            ari = adjusted_rand_score(base_labels[:len(week_labels)], week_labels[:len(base_labels)])
            results.append({
                'week': week,
                'date': week_date,
                'ari': ari,
                'n_customers': len(week_data)
            })

    stability_df = pd.DataFrame(results)
    logger.info(f"Temporal stability analysis complete: {len(stability_df)} weeks")

    return stability_df


if __name__ == "__main__":
    # Example usage
    from src.data.data_loader import load_processed_data
    from src.models.clustering import CustomerSegmenter

    # Load data
    rfm_data = load_processed_data("data/processed/data_RFM.csv")

    # Train model
    segmenter = CustomerSegmenter(algorithm='kmeans', n_clusters=4, random_state=42)
    labels = segmenter.fit_predict(rfm_data, feature_columns=['Recency', 'Frequency', 'Monetary'])

    # Evaluate
    metrics = evaluate_clustering(rfm_data, labels, ['Recency', 'Frequency', 'Monetary'])
    print(f"\nEvaluation Metrics: {metrics}")
