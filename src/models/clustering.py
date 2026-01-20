"""
Customer clustering models for segmentation.

This module implements various clustering algorithms:
- K-Means
- DBSCAN
- Agglomerative Clustering
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from typing import Dict, Optional, Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomerSegmenter:
    """
    Performs customer segmentation using clustering algorithms.

    Attributes:
        model: The clustering model.
        scaler: StandardScaler for feature normalization.
        feature_columns: List of feature column names.
        labels_: Cluster labels for training data.
    """

    def __init__(self, algorithm: str = 'kmeans', **kwargs):
        """
        Initialize the segmenter.

        Parameters:
            algorithm (str): Clustering algorithm ('kmeans', 'dbscan', 'agglomerative').
            **kwargs: Additional parameters for the clustering algorithm.
        """
        self.algorithm = algorithm
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.labels_ = None
        self.model = self._initialize_model(**kwargs)

    def _initialize_model(self, **kwargs):
        """Initialize the clustering model based on algorithm choice."""
        if self.algorithm == 'kmeans':
            n_clusters = kwargs.get('n_clusters', 4)
            random_state = kwargs.get('random_state', 42)
            return KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=10
            )
        elif self.algorithm == 'dbscan':
            eps = kwargs.get('eps', 0.5)
            min_samples = kwargs.get('min_samples', 5)
            return DBSCAN(eps=eps, min_samples=min_samples)
        elif self.algorithm == 'agglomerative':
            n_clusters = kwargs.get('n_clusters', 4)
            linkage = kwargs.get('linkage', 'ward')
            return AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def fit(
        self,
        data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None
    ) -> 'CustomerSegmenter':
        """
        Fit the clustering model.

        Parameters:
            data (pd.DataFrame): Customer data.
            feature_columns (List[str], optional): Columns to use for clustering.
                If None, uses ['Recency', 'Frequency', 'Monetary'].

        Returns:
            CustomerSegmenter: Fitted model.
        """
        if feature_columns is None:
            feature_columns = ['Recency', 'Frequency', 'Monetary']

        self.feature_columns = feature_columns
        logger.info(f"Fitting {self.algorithm} with features: {feature_columns}")

        # Extract features
        X = data[feature_columns].values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit model
        self.model.fit(X_scaled)
        self.labels_ = self.model.labels_

        logger.info(f"Clustering complete. Found {len(np.unique(self.labels_))} clusters")
        return self

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict cluster labels for new data.

        Parameters:
            data (pd.DataFrame): Customer data.

        Returns:
            np.ndarray: Cluster labels.
        """
        if self.feature_columns is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = data[self.feature_columns].values
        X_scaled = self.scaler.transform(X)

        # For DBSCAN and Agglomerative, we can't predict on new data
        # Use KMeans predict or fit_predict for all data
        if hasattr(self.model, 'predict'):
            labels = self.model.predict(X_scaled)
        else:
            logger.warning(f"{self.algorithm} doesn't support predict(). Use fit() instead.")
            labels = None

        return labels

    def fit_predict(self, data: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> np.ndarray:
        """
        Fit the model and return cluster labels.

        Parameters:
            data (pd.DataFrame): Customer data.
            feature_columns (List[str], optional): Columns to use for clustering.

        Returns:
            np.ndarray: Cluster labels.
        """
        self.fit(data, feature_columns)
        return self.labels_

    def get_cluster_profiles(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get descriptive statistics for each cluster.

        Parameters:
            data (pd.DataFrame): Customer data with cluster labels.

        Returns:
            pd.DataFrame: Cluster profiles.
        """
        if self.labels_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        df = data.copy()
        df['Cluster'] = self.labels_

        # Calculate profiles
        profiles = df.groupby('Cluster')[self.feature_columns].agg(['mean', 'median', 'std', 'count'])

        logger.info("\nCluster Profiles:")
        logger.info(f"\n{profiles}")

        return profiles

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.

        Parameters:
            filepath (str): Path to save the model.
        """
        model_data = {
            'algorithm': self.algorithm,
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'labels_': self.labels_
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> 'CustomerSegmenter':
        """
        Load a trained model from disk.

        Parameters:
            filepath (str): Path to the saved model.

        Returns:
            CustomerSegmenter: Loaded model.
        """
        model_data = joblib.load(filepath)

        segmenter = cls(algorithm=model_data['algorithm'])
        segmenter.model = model_data['model']
        segmenter.scaler = model_data['scaler']
        segmenter.feature_columns = model_data['feature_columns']
        segmenter.labels_ = model_data['labels_']

        logger.info(f"Model loaded from {filepath}")
        return segmenter


def find_optimal_k(
    data: pd.DataFrame,
    feature_columns: List[str],
    k_range: range = range(2, 11),
    random_state: int = 42
) -> Tuple[List[int], List[float]]:
    """
    Find optimal number of clusters using elbow method (inertia).

    Parameters:
        data (pd.DataFrame): Customer data.
        feature_columns (List[str]): Features for clustering.
        k_range (range): Range of k values to test.
        random_state (int): Random state for reproducibility.

    Returns:
        Tuple[List[int], List[float]]: K values and corresponding inertias.
    """
    logger.info(f"Testing k values from {k_range.start} to {k_range.stop - 1}...")

    scaler = StandardScaler()
    X = data[feature_columns].values
    X_scaled = scaler.fit_transform(X)

    inertias = []
    k_values = list(k_range)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        logger.info(f"k={k}: inertia={kmeans.inertia_:.2f}")

    return k_values, inertias


if __name__ == "__main__":
    # Example usage
    from src.data.data_loader import load_processed_data

    # Load RFM data
    rfm_data = load_processed_data("data/processed/data_RFM.csv")

    # Create and fit K-Means model
    segmenter = CustomerSegmenter(algorithm='kmeans', n_clusters=4, random_state=42)
    labels = segmenter.fit_predict(rfm_data, feature_columns=['Recency', 'Frequency', 'Monetary'])

    # Get cluster profiles
    rfm_data['Cluster'] = labels
    profiles = segmenter.get_cluster_profiles(rfm_data)

    # Save model
    segmenter.save_model("outputs/models/kmeans_segmentation.pkl")
