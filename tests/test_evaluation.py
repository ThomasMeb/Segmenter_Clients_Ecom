"""Tests pour le module d'évaluation des modèles."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from src.models.evaluation import (
    calculate_calinski_harabasz_score,
    calculate_silhouette_score,
    compare_models,
    evaluate_clustering,
    get_silhouette_per_sample,
)


@pytest.fixture
def clustered_data():
    """Crée des données avec des clusters bien séparés."""
    X, y = make_blobs(
        n_samples=300,
        n_features=3,
        centers=4,
        cluster_std=0.5,
        random_state=42,
    )
    return X, y


@pytest.fixture
def poorly_clustered_data():
    """Crée des données avec des clusters mal séparés."""
    X, y = make_blobs(
        n_samples=300,
        n_features=3,
        centers=4,
        cluster_std=3.0,  # Grande variance = clusters qui se chevauchent
        random_state=42,
    )
    return X, y


class TestEvaluateClustering:
    """Tests pour la fonction evaluate_clustering."""

    def test_returns_dict(self, clustered_data):
        """Test que la fonction retourne un dictionnaire."""
        X, labels = clustered_data
        result = evaluate_clustering(X, labels)
        assert isinstance(result, dict)

    def test_contains_required_keys(self, clustered_data):
        """Test que le dictionnaire contient les clés requises."""
        X, labels = clustered_data
        result = evaluate_clustering(X, labels)
        required_keys = [
            "silhouette",
            "calinski_harabasz",
            "davies_bouldin",
            "n_clusters",
        ]
        for key in required_keys:
            assert key in result

    def test_silhouette_range(self, clustered_data):
        """Test que le silhouette score est dans [-1, 1]."""
        X, labels = clustered_data
        result = evaluate_clustering(X, labels)
        assert -1 <= result["silhouette"] <= 1

    def test_calinski_harabasz_positive(self, clustered_data):
        """Test que le Calinski-Harabasz est positif."""
        X, labels = clustered_data
        result = evaluate_clustering(X, labels)
        assert result["calinski_harabasz"] > 0

    def test_davies_bouldin_positive(self, clustered_data):
        """Test que le Davies-Bouldin est positif."""
        X, labels = clustered_data
        result = evaluate_clustering(X, labels)
        assert result["davies_bouldin"] > 0

    def test_n_clusters_correct(self, clustered_data):
        """Test que le nombre de clusters est correct."""
        X, labels = clustered_data
        result = evaluate_clustering(X, labels)
        assert result["n_clusters"] == len(np.unique(labels))

    def test_good_clustering_has_high_silhouette(self, clustered_data):
        """Test qu'un bon clustering a un silhouette élevé."""
        X, labels = clustered_data
        result = evaluate_clustering(X, labels)
        assert result["silhouette"] > 0.5  # Clusters bien séparés

    def test_poor_clustering_has_lower_silhouette(self, poorly_clustered_data):
        """Test qu'un mauvais clustering a un silhouette plus bas."""
        X, labels = poorly_clustered_data
        result = evaluate_clustering(X, labels)
        assert result["silhouette"] < 0.5  # Clusters mal séparés


class TestCalculateSilhouetteScore:
    """Tests pour la fonction calculate_silhouette_score."""

    def test_returns_float(self, clustered_data):
        """Test que la fonction retourne un float."""
        X, labels = clustered_data
        result = calculate_silhouette_score(X, labels)
        assert isinstance(result, float)

    def test_score_in_valid_range(self, clustered_data):
        """Test que le score est dans [-1, 1]."""
        X, labels = clustered_data
        result = calculate_silhouette_score(X, labels)
        assert -1 <= result <= 1

    def test_consistent_with_evaluate_clustering(self, clustered_data):
        """Test de cohérence avec evaluate_clustering."""
        X, labels = clustered_data
        silhouette = calculate_silhouette_score(X, labels)
        full_eval = evaluate_clustering(X, labels)
        assert silhouette == full_eval["silhouette"]


class TestCalculateCalinskiHarabaszScore:
    """Tests pour la fonction calculate_calinski_harabasz_score."""

    def test_returns_float(self, clustered_data):
        """Test que la fonction retourne un float."""
        X, labels = clustered_data
        result = calculate_calinski_harabasz_score(X, labels)
        assert isinstance(result, float)

    def test_score_positive(self, clustered_data):
        """Test que le score est positif."""
        X, labels = clustered_data
        result = calculate_calinski_harabasz_score(X, labels)
        assert result > 0

    def test_consistent_with_evaluate_clustering(self, clustered_data):
        """Test de cohérence avec evaluate_clustering."""
        X, labels = clustered_data
        ch_score = calculate_calinski_harabasz_score(X, labels)
        full_eval = evaluate_clustering(X, labels)
        assert ch_score == full_eval["calinski_harabasz"]


class TestGetSilhouettePerSample:
    """Tests pour la fonction get_silhouette_per_sample."""

    def test_returns_dataframe(self, clustered_data):
        """Test que la fonction retourne un DataFrame."""
        X, labels = clustered_data
        result = get_silhouette_per_sample(X, labels)
        assert isinstance(result, pd.DataFrame)

    def test_contains_required_columns(self, clustered_data):
        """Test que le DataFrame contient les colonnes requises."""
        X, labels = clustered_data
        result = get_silhouette_per_sample(X, labels)
        assert "cluster" in result.columns
        assert "silhouette" in result.columns

    def test_length_matches_input(self, clustered_data):
        """Test que la longueur correspond à l'input."""
        X, labels = clustered_data
        result = get_silhouette_per_sample(X, labels)
        assert len(result) == len(labels)

    def test_silhouette_values_in_range(self, clustered_data):
        """Test que les scores individuels sont dans [-1, 1]."""
        X, labels = clustered_data
        result = get_silhouette_per_sample(X, labels)
        assert (result["silhouette"] >= -1).all()
        assert (result["silhouette"] <= 1).all()

    def test_cluster_labels_match(self, clustered_data):
        """Test que les labels de cluster correspondent."""
        X, labels = clustered_data
        result = get_silhouette_per_sample(X, labels)
        np.testing.assert_array_equal(result["cluster"].values, labels)


class TestCompareModels:
    """Tests pour la fonction compare_models."""

    @pytest.fixture
    def multiple_clusterings(self, clustered_data):
        """Crée plusieurs clusterings à comparer."""
        X, labels_4 = clustered_data
        # Créer un clustering avec 3 clusters (fusionner 2 clusters)
        labels_3 = labels_4.copy()
        labels_3[labels_3 == 3] = 0
        return X, {"KMeans k=4": labels_4, "KMeans k=3": labels_3}

    def test_returns_dataframe(self, multiple_clusterings):
        """Test que la fonction retourne un DataFrame."""
        X, models_labels = multiple_clusterings
        result = compare_models(X, models_labels)
        assert isinstance(result, pd.DataFrame)

    def test_index_is_model_names(self, multiple_clusterings):
        """Test que l'index contient les noms des modèles."""
        X, models_labels = multiple_clusterings
        result = compare_models(X, models_labels)
        assert list(result.index) == list(models_labels.keys())

    def test_contains_metrics_columns(self, multiple_clusterings):
        """Test que le DataFrame contient les colonnes de métriques."""
        X, models_labels = multiple_clusterings
        result = compare_models(X, models_labels)
        expected_cols = [
            "silhouette",
            "calinski_harabasz",
            "davies_bouldin",
            "n_clusters",
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_values_are_rounded(self, multiple_clusterings):
        """Test que les valeurs sont arrondies à 4 décimales."""
        X, models_labels = multiple_clusterings
        result = compare_models(X, models_labels)
        # Vérifier que les valeurs ont au maximum 4 décimales
        for col in ["silhouette", "calinski_harabasz", "davies_bouldin"]:
            for val in result[col]:
                str_val = str(val)
                if "." in str_val:
                    decimals = len(str_val.split(".")[1])
                    assert decimals <= 4

    def test_n_clusters_matches(self, multiple_clusterings):
        """Test que n_clusters correspond au nombre réel de clusters."""
        X, models_labels = multiple_clusterings
        result = compare_models(X, models_labels)
        assert result.loc["KMeans k=4", "n_clusters"] == 4
        assert result.loc["KMeans k=3", "n_clusters"] == 3

    def test_single_model_comparison(self, clustered_data):
        """Test avec un seul modèle."""
        X, labels = clustered_data
        result = compare_models(X, {"single_model": labels})
        assert len(result) == 1
        assert "single_model" in result.index
