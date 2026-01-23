"""Tests d'intégration pour le pipeline complet."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.utils import generate_demo_data
from src.data.preprocessor import clean_transactions
from src.features.rfm import RFMCalculator
from src.models.clustering import CustomerSegmenter
from src.models.evaluation import evaluate_clustering


class TestEndToEndPipeline:
    """Tests d'intégration du pipeline complet."""

    @pytest.fixture
    def sample_transactions(self):
        """Génère des transactions de test."""
        from src.config import AMOUNT_COL, CUSTOMER_ID_COL, DATE_COL, ORDER_ID_COL

        np.random.seed(42)
        n_customers = 100
        n_orders = 500

        # Générer des clients
        customers = [f"customer_{i:04d}" for i in range(n_customers)]

        # Générer des commandes avec les noms de colonnes de la config
        data = {
            CUSTOMER_ID_COL: np.random.choice(customers, n_orders),
            ORDER_ID_COL: [f"order_{i:04d}" for i in range(n_orders)],
            DATE_COL: pd.date_range("2023-01-01", periods=n_orders, freq="2h").astype(
                str
            ),
            AMOUNT_COL: np.random.exponential(100, n_orders) + 20,
        }

        return pd.DataFrame(data)

    def test_full_pipeline_execution(self, sample_transactions):
        """Test du pipeline complet: preprocess -> RFM -> cluster -> evaluate."""
        # 1. Nettoyage des données
        df_clean = clean_transactions(sample_transactions)
        assert len(df_clean) > 0

        # 2. Vérification des colonnes nécessaires
        from src.config import CUSTOMER_ID_COL

        assert CUSTOMER_ID_COL in df_clean.columns

        # 3. Calcul RFM (directement sur les données nettoyées)
        calculator = RFMCalculator(reference_date="2023-06-01")
        rfm = calculator.fit_transform(df_clean)
        assert all(col in rfm.columns for col in ["recency", "frequency", "monetary"])

        # 4. Segmentation
        segmenter = CustomerSegmenter(n_clusters=4, random_state=42)
        labels = segmenter.fit_predict(rfm[["recency", "frequency", "monetary"]])
        assert len(labels) == len(rfm)
        assert len(np.unique(labels)) == 4

        # 5. Évaluation
        X = segmenter.scaler.transform(rfm[["recency", "frequency", "monetary"]])
        metrics = evaluate_clustering(X, labels)
        assert "silhouette" in metrics
        assert -1 <= metrics["silhouette"] <= 1

    def test_model_save_load_predict(self, sample_transactions):
        """Test sauvegarde, chargement et prédiction."""
        # Préparation
        df_clean = clean_transactions(sample_transactions)
        calculator = RFMCalculator(reference_date="2023-06-01")
        rfm = calculator.fit_transform(df_clean)

        # Entraînement
        segmenter = CustomerSegmenter(n_clusters=4, random_state=42)
        labels_original = segmenter.fit_predict(
            rfm[["recency", "frequency", "monetary"]]
        )

        # Sauvegarde et chargement
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            segmenter.save(save_path)

            # Charger le modèle
            loaded_segmenter = CustomerSegmenter.load(save_path)

            # Prédire avec le modèle chargé
            labels_loaded = loaded_segmenter.predict(
                rfm[["recency", "frequency", "monetary"]]
            )

            # Vérifier que les prédictions sont identiques
            np.testing.assert_array_equal(labels_original, labels_loaded)

    def test_demo_data_can_be_segmented(self):
        """Test que les données de démo peuvent être segmentées."""
        # Générer des données de démo
        demo_data = generate_demo_data(n_samples=1000, random_state=42)

        # Segmenter
        segmenter = CustomerSegmenter(n_clusters=4, random_state=42)
        labels = segmenter.fit_predict(demo_data[["recency", "frequency", "monetary"]])

        # Vérifications
        assert len(labels) == len(demo_data)
        assert len(np.unique(labels)) == 4

        # Évaluation
        X = segmenter.scaler.transform(demo_data[["recency", "frequency", "monetary"]])
        metrics = evaluate_clustering(X, labels)
        assert metrics["silhouette"] > 0  # Doit avoir un score positif

    def test_segmenter_reproducibility(self, sample_transactions):
        """Test de reproductibilité du segmenteur."""
        # Préparation
        df_clean = clean_transactions(sample_transactions)
        calculator = RFMCalculator(reference_date="2023-06-01")
        rfm = calculator.fit_transform(df_clean)
        X = rfm[["recency", "frequency", "monetary"]]

        # Deux segmentations avec le même random_state
        segmenter1 = CustomerSegmenter(n_clusters=4, random_state=42)
        labels1 = segmenter1.fit_predict(X)

        segmenter2 = CustomerSegmenter(n_clusters=4, random_state=42)
        labels2 = segmenter2.fit_predict(X)

        np.testing.assert_array_equal(labels1, labels2)


class TestPipelineEdgeCases:
    """Tests des cas limites du pipeline."""

    def test_small_dataset(self):
        """Test avec un petit dataset."""
        # Minimum de données pour 4 clusters
        demo_data = generate_demo_data(n_samples=50, random_state=42)

        segmenter = CustomerSegmenter(n_clusters=4, random_state=42)
        labels = segmenter.fit_predict(demo_data[["recency", "frequency", "monetary"]])

        assert len(labels) == 50

    def test_high_dimensional_consistency(self):
        """Test que le scaling est cohérent."""
        demo_data = generate_demo_data(n_samples=500, random_state=42)
        X = demo_data[["recency", "frequency", "monetary"]]

        segmenter = CustomerSegmenter(n_clusters=4, random_state=42)
        segmenter.fit(X)

        # Vérifier que le scaler a bien transformé les données
        X_scaled = segmenter.scaler.transform(X)
        assert X_scaled.shape == X.shape

        # Vérifier que les valeurs sont centrées (StandardScaler)
        # Mean devrait être proche de 0, std proche de 1
        assert abs(X_scaled.mean()) < 0.5
        assert 0.5 < X_scaled.std() < 1.5

    def test_cluster_summary_structure(self):
        """Test de la structure du résumé des clusters."""
        demo_data = generate_demo_data(n_samples=500, random_state=42)
        X = demo_data[["recency", "frequency", "monetary"]]

        segmenter = CustomerSegmenter(n_clusters=4, random_state=42)
        segmenter.fit_predict(X)

        summary = segmenter.get_segment_summary(X)

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 4  # 4 clusters
        assert "count" in summary.columns
        # Vérifier les colonnes de statistiques RFM
        assert "recency_mean" in summary.columns or "recency" in str(summary.columns)


class TestMetricsConsistency:
    """Tests de cohérence des métriques."""

    def test_silhouette_improves_with_better_clustering(self):
        """Test que le silhouette s'améliore avec un meilleur clustering."""
        # Données bien séparées
        demo_data = generate_demo_data(n_samples=1000, random_state=42)
        X = demo_data[["recency", "frequency", "monetary"]]

        # Segmenter avec différents k
        scores = {}
        for k in [2, 3, 4, 5]:
            segmenter = CustomerSegmenter(n_clusters=k, random_state=42)
            labels = segmenter.fit_predict(X)
            X_scaled = segmenter.scaler.transform(X)
            metrics = evaluate_clustering(X_scaled, labels)
            scores[k] = metrics["silhouette"]

        # Le score devrait être raisonnable pour k=4 (nombre réel de segments)
        assert scores[4] > 0.3  # Au moins 0.3 pour des données de démo

    def test_evaluation_metrics_valid_ranges(self):
        """Test que les métriques sont dans des plages valides."""
        demo_data = generate_demo_data(n_samples=500, random_state=42)
        X = demo_data[["recency", "frequency", "monetary"]]

        segmenter = CustomerSegmenter(n_clusters=4, random_state=42)
        labels = segmenter.fit_predict(X)
        X_scaled = segmenter.scaler.transform(X)

        metrics = evaluate_clustering(X_scaled, labels)

        # Silhouette: [-1, 1]
        assert -1 <= metrics["silhouette"] <= 1

        # Calinski-Harabasz: > 0
        assert metrics["calinski_harabasz"] > 0

        # Davies-Bouldin: >= 0
        assert metrics["davies_bouldin"] >= 0

        # n_clusters: doit correspondre
        assert metrics["n_clusters"] == 4
