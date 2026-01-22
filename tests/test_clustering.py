"""
Tests pour le module src.models.clustering.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.models.clustering import CustomerSegmenter, find_optimal_clusters
from src.config import SEGMENT_NAMES


class TestCustomerSegmenter:
    """Tests pour la classe CustomerSegmenter."""

    def test_init_default(self):
        """Test d'initialisation avec paramètres par défaut."""
        segmenter = CustomerSegmenter()

        assert segmenter.n_clusters == 4
        assert segmenter.random_state == 42
        assert segmenter.is_fitted is False

    def test_init_custom_params(self):
        """Test d'initialisation avec paramètres personnalisés."""
        segmenter = CustomerSegmenter(n_clusters=5, random_state=123)

        assert segmenter.n_clusters == 5
        assert segmenter.random_state == 123

    def test_fit(self, sample_rfm):
        """Test de la méthode fit."""
        segmenter = CustomerSegmenter(n_clusters=3)
        result = segmenter.fit(sample_rfm)

        # Vérifier le chaînage
        assert result is segmenter
        assert segmenter.is_fitted is True

    def test_predict(self, sample_rfm):
        """Test de la méthode predict."""
        segmenter = CustomerSegmenter(n_clusters=3)
        segmenter.fit(sample_rfm)

        labels = segmenter.predict(sample_rfm)

        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(sample_rfm)
        assert set(labels).issubset({0, 1, 2})

    def test_predict_without_fit(self, sample_rfm):
        """Test que predict lève une erreur sans fit."""
        segmenter = CustomerSegmenter()

        with pytest.raises(ValueError, match="entraîné"):
            segmenter.predict(sample_rfm)

    def test_fit_predict(self, sample_rfm):
        """Test de la méthode fit_predict."""
        segmenter = CustomerSegmenter(n_clusters=3)
        labels = segmenter.fit_predict(sample_rfm)

        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(sample_rfm)
        assert segmenter.is_fitted is True

    def test_get_cluster_centers_original(self, sample_rfm):
        """Test des centres de clusters en coordonnées originales."""
        segmenter = CustomerSegmenter(n_clusters=3)
        segmenter.fit(sample_rfm)

        centers = segmenter.get_cluster_centers(scaled=False)

        assert isinstance(centers, pd.DataFrame)
        assert len(centers) == 3
        assert list(centers.columns) == list(sample_rfm.columns)

    def test_get_cluster_centers_scaled(self, sample_rfm):
        """Test des centres de clusters en coordonnées scalées."""
        segmenter = CustomerSegmenter(n_clusters=3)
        segmenter.fit(sample_rfm)

        centers = segmenter.get_cluster_centers(scaled=True)

        assert isinstance(centers, pd.DataFrame)
        assert len(centers) == 3

    def test_get_cluster_centers_without_fit(self, sample_rfm):
        """Test que get_cluster_centers lève une erreur sans fit."""
        segmenter = CustomerSegmenter()

        with pytest.raises(ValueError, match="entraîné"):
            segmenter.get_cluster_centers()

    def test_get_segment_summary(self, sample_rfm):
        """Test du résumé des segments."""
        segmenter = CustomerSegmenter(n_clusters=4)
        segmenter.fit(sample_rfm)

        summary = segmenter.get_segment_summary(sample_rfm)

        assert isinstance(summary, pd.DataFrame)
        assert "count" in summary.columns
        assert summary["count"].sum() == len(sample_rfm)

    def test_save_and_load(self, sample_rfm, test_data_dir):
        """Test de sauvegarde et chargement du modèle."""
        # Entraîner et sauvegarder
        segmenter = CustomerSegmenter(n_clusters=3)
        segmenter.fit(sample_rfm)
        original_labels = segmenter.predict(sample_rfm)

        model_dir = test_data_dir / "models"
        model_dir.mkdir()
        model_path, scaler_path = segmenter.save(model_dir)

        # Vérifier que les fichiers existent
        assert model_path.exists()
        assert scaler_path.exists()

        # Charger et vérifier
        loaded_segmenter = CustomerSegmenter.load(model_dir)
        loaded_labels = loaded_segmenter.predict(sample_rfm)

        # Les prédictions doivent être identiques
        np.testing.assert_array_equal(original_labels, loaded_labels)

    def test_save_without_fit(self, test_data_dir):
        """Test que save lève une erreur sans fit."""
        segmenter = CustomerSegmenter()

        with pytest.raises(ValueError, match="entraîné"):
            segmenter.save(test_data_dir)

    def test_reproducibility(self, sample_rfm):
        """Test de la reproductibilité avec random_state."""
        segmenter1 = CustomerSegmenter(n_clusters=3, random_state=42)
        labels1 = segmenter1.fit_predict(sample_rfm)

        segmenter2 = CustomerSegmenter(n_clusters=3, random_state=42)
        labels2 = segmenter2.fit_predict(sample_rfm)

        np.testing.assert_array_equal(labels1, labels2)


class TestFindOptimalClusters:
    """Tests pour la fonction find_optimal_clusters."""

    def test_find_optimal_clusters_basic(self, sample_rfm):
        """Test de base de find_optimal_clusters."""
        results = find_optimal_clusters(sample_rfm, k_range=range(2, 6))

        assert isinstance(results, pd.DataFrame)
        assert "k" in results.columns
        assert "inertia" in results.columns
        assert "silhouette" in results.columns

    def test_find_optimal_clusters_range(self, sample_rfm):
        """Test que le range de k est respecté."""
        k_range = range(2, 5)
        results = find_optimal_clusters(sample_rfm, k_range=k_range)

        assert len(results) == len(list(k_range))
        assert list(results["k"]) == list(k_range)

    def test_find_optimal_clusters_inertia_decreasing(self, sample_rfm):
        """Test que l'inertie diminue avec k."""
        results = find_optimal_clusters(sample_rfm, k_range=range(2, 6))

        # L'inertie doit généralement diminuer
        inertias = results["inertia"].values
        # Au moins la tendance générale doit être décroissante
        assert inertias[0] > inertias[-1]

    def test_find_optimal_clusters_silhouette_valid(self, sample_rfm):
        """Test que les scores de silhouette sont valides."""
        results = find_optimal_clusters(sample_rfm, k_range=range(2, 6))

        # Silhouette doit être entre -1 et 1
        assert (results["silhouette"] >= -1).all()
        assert (results["silhouette"] <= 1).all()

    def test_find_optimal_clusters_reproducibility(self, sample_rfm):
        """Test de la reproductibilité."""
        results1 = find_optimal_clusters(sample_rfm, k_range=range(2, 5), random_state=42)
        results2 = find_optimal_clusters(sample_rfm, k_range=range(2, 5), random_state=42)

        pd.testing.assert_frame_equal(results1, results2)
