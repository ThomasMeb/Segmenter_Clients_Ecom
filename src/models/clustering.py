"""
Module de clustering pour la segmentation client.

Ce module implémente les modèles de clustering utilisés pour
segmenter les clients selon leurs features RFM.
"""

from pathlib import Path
from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.config import (
    N_CLUSTERS,
    RANDOM_STATE,
    KMEANS_N_INIT,
    KMEANS_MAX_ITER,
    KMEANS_MODEL_FILE,
    SCALER_FILE,
    SEGMENT_NAMES,
    RFM_FEATURES,
)


class CustomerSegmenter:
    """
    Pipeline de segmentation client basé sur KMeans.

    Cette classe encapsule le preprocessing (scaling) et le modèle
    de clustering pour une utilisation simplifiée.

    Attributes
    ----------
    n_clusters : int
        Nombre de clusters (segments).
    scaler : StandardScaler
        Scaler pour la normalisation des features.
    model : KMeans
        Modèle KMeans entraîné.
    is_fitted : bool
        Indique si le modèle a été entraîné.

    Examples
    --------
    >>> segmenter = CustomerSegmenter(n_clusters=4)
    >>> segmenter.fit(rfm_df)
    >>> labels = segmenter.predict(rfm_df)
    >>> segmenter.save("models/")
    """

    def __init__(
        self,
        n_clusters: int = N_CLUSTERS,
        random_state: int = RANDOM_STATE,
    ):
        """
        Initialise le segmenter.

        Parameters
        ----------
        n_clusters : int, default 4
            Nombre de segments à créer.
        random_state : int, default 42
            Graine aléatoire pour la reproductibilité.
        """
        self.n_clusters = n_clusters
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=KMEANS_N_INIT,
            max_iter=KMEANS_MAX_ITER,
        )

        self.is_fitted = False
        self._feature_names: Optional[list] = None

    def fit(self, X: pd.DataFrame) -> "CustomerSegmenter":
        """
        Entraîne le modèle de segmentation.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame avec les features RFM.

        Returns
        -------
        CustomerSegmenter
            Instance entraînée (pour le chaînage).
        """
        self._feature_names = X.columns.tolist()

        # Scaling
        X_scaled = self.scaler.fit_transform(X)

        # Clustering
        self.model.fit(X_scaled)

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prédit les segments pour de nouveaux clients.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame avec les features RFM.

        Returns
        -------
        np.ndarray
            Labels des segments (0 à n_clusters-1).

        Raises
        ------
        ValueError
            Si le modèle n'a pas été entraîné.
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné. Appelez fit() d'abord.")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def fit_predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Entraîne le modèle et retourne les prédictions.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame avec les features RFM.

        Returns
        -------
        np.ndarray
            Labels des segments.
        """
        self.fit(X)
        return self.model.labels_

    def get_cluster_centers(self, scaled: bool = False) -> pd.DataFrame:
        """
        Retourne les centres des clusters.

        Parameters
        ----------
        scaled : bool, default False
            Si True, retourne les centres en coordonnées scalées.

        Returns
        -------
        pd.DataFrame
            Centres des clusters avec les noms de features.
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné.")

        centers = self.model.cluster_centers_

        if not scaled:
            centers = self.scaler.inverse_transform(centers)

        return pd.DataFrame(
            centers,
            columns=self._feature_names,
            index=[f"Cluster {i}" for i in range(self.n_clusters)],
        )

    def get_segment_summary(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Retourne un résumé des segments.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame avec les features RFM.

        Returns
        -------
        pd.DataFrame
            Statistiques par segment (mean, count, etc.).
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné.")

        df = X.copy()
        df["segment"] = self.predict(X)
        df["segment_name"] = df["segment"].map(SEGMENT_NAMES)

        summary = df.groupby("segment_name").agg({
            "recency": ["mean", "std"],
            "frequency": ["mean", "std"],
            "monetary": ["mean", "std"],
            "segment": "count",
        }).round(2)

        summary.columns = [
            "recency_mean", "recency_std",
            "frequency_mean", "frequency_std",
            "monetary_mean", "monetary_std",
            "count",
        ]

        return summary

    def save(self, directory: Optional[Union[str, Path]] = None) -> Tuple[Path, Path]:
        """
        Sauvegarde le modèle et le scaler.

        Parameters
        ----------
        directory : str or Path, optional
            Dossier de destination. Par défaut, utilise models/.

        Returns
        -------
        tuple of Path
            Chemins des fichiers sauvegardés (model, scaler).
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné.")

        if directory is None:
            model_path = KMEANS_MODEL_FILE
            scaler_path = SCALER_FILE
        else:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            model_path = directory / "kmeans_model.pkl"
            scaler_path = directory / "scaler.pkl"

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

        return model_path, scaler_path

    @classmethod
    def load(
        cls,
        directory: Optional[Union[str, Path]] = None,
    ) -> "CustomerSegmenter":
        """
        Charge un modèle sauvegardé.

        Parameters
        ----------
        directory : str or Path, optional
            Dossier contenant les fichiers. Par défaut, utilise models/.

        Returns
        -------
        CustomerSegmenter
            Instance avec le modèle chargé.
        """
        if directory is None:
            model_path = KMEANS_MODEL_FILE
            scaler_path = SCALER_FILE
        else:
            directory = Path(directory)
            model_path = directory / "kmeans_model.pkl"
            scaler_path = directory / "scaler.pkl"

        instance = cls()
        instance.model = joblib.load(model_path)
        instance.scaler = joblib.load(scaler_path)
        instance.n_clusters = instance.model.n_clusters
        instance.is_fitted = True

        return instance


def find_optimal_clusters(
    X: pd.DataFrame,
    k_range: range = range(2, 11),
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Trouve le nombre optimal de clusters en utilisant la méthode du coude.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame avec les features.
    k_range : range, default range(2, 11)
        Plage de valeurs de k à tester.
    random_state : int, default 42
        Graine aléatoire.

    Returns
    -------
    pd.DataFrame
        DataFrame avec k, inertia, et silhouette score.
    """
    from sklearn.metrics import silhouette_score

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = []

    for k in k_range:
        kmeans = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=KMEANS_N_INIT,
        )
        labels = kmeans.fit_predict(X_scaled)

        results.append({
            "k": k,
            "inertia": kmeans.inertia_,
            "silhouette": silhouette_score(X_scaled, labels),
        })

    return pd.DataFrame(results)
