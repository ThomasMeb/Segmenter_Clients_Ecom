"""Tests pour le module app/utils.py."""

from pathlib import Path

import pandas as pd
import pytest

from app.utils import (
    generate_demo_data,
    get_project_root,
    load_rfm_data,
)


class TestGetProjectRoot:
    """Tests pour la fonction get_project_root."""

    def test_returns_path(self):
        """Test que la fonction retourne un Path."""
        result = get_project_root()
        assert isinstance(result, Path)

    def test_path_exists(self):
        """Test que le chemin retourné existe."""
        result = get_project_root()
        assert result.exists()

    def test_contains_expected_files(self):
        """Test que le chemin contient des fichiers attendus du projet."""
        root = get_project_root()
        # Le projet doit avoir ces fichiers/dossiers
        assert (root / "app").exists()
        assert (root / "src").exists()


class TestGenerateDemoData:
    """Tests pour la fonction generate_demo_data."""

    def test_returns_dataframe(self):
        """Test que la fonction retourne un DataFrame."""
        result = generate_demo_data()
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self):
        """Test que le DataFrame a les colonnes requises."""
        result = generate_demo_data()
        required_cols = ["recency", "frequency", "monetary", "segment"]
        for col in required_cols:
            assert col in result.columns

    def test_respects_n_samples(self):
        """Test que n_samples est respecté."""
        n = 500
        result = generate_demo_data(n_samples=n)
        assert len(result) == n

    def test_reproducibility(self):
        """Test de reproductibilité avec random_state."""
        df1 = generate_demo_data(n_samples=100, random_state=42)
        df2 = generate_demo_data(n_samples=100, random_state=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_random_states_differ(self):
        """Test que différents random_state donnent des résultats différents."""
        df1 = generate_demo_data(n_samples=100, random_state=42)
        df2 = generate_demo_data(n_samples=100, random_state=123)
        assert not df1.equals(df2)

    def test_segment_distribution(self):
        """Test que la distribution des segments est approximativement correcte."""
        df = generate_demo_data(n_samples=10000, random_state=42)
        segment_counts = df["segment"].value_counts(normalize=True)

        # Segment 0 (Recent): ~54%
        assert 0.50 <= segment_counts.get(0, 0) <= 0.58
        # Segment 2 (Dormant): ~40%
        assert 0.36 <= segment_counts.get(2, 0) <= 0.44

    def test_recency_positive(self):
        """Test que la récence est positive."""
        df = generate_demo_data(n_samples=1000)
        assert (df["recency"] > 0).all()

    def test_frequency_positive(self):
        """Test que la fréquence est positive."""
        df = generate_demo_data(n_samples=1000)
        assert (df["frequency"] > 0).all()

    def test_monetary_positive(self):
        """Test que le montant est positif."""
        df = generate_demo_data(n_samples=1000)
        assert (df["monetary"] > 0).all()

    def test_has_customer_index(self):
        """Test que l'index contient des IDs client."""
        df = generate_demo_data(n_samples=100)
        assert df.index.name == "customer_unique_id"
        assert df.index[0].startswith("customer_")

    def test_segments_are_valid(self):
        """Test que tous les segments sont valides (0-3)."""
        df = generate_demo_data(n_samples=1000)
        assert df["segment"].isin([0, 1, 2, 3]).all()


class TestLoadRfmData:
    """Tests pour la fonction load_rfm_data."""

    def test_returns_tuple(self):
        """Test que la fonction retourne un tuple."""
        result = load_rfm_data()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_tuple_contains_dataframe_and_bool(self):
        """Test que le tuple contient (DataFrame, bool)."""
        df, is_real = load_rfm_data()
        assert isinstance(df, pd.DataFrame)
        assert isinstance(is_real, bool)

    def test_dataframe_has_required_columns(self):
        """Test que le DataFrame a les colonnes requises."""
        df, _ = load_rfm_data()
        required_cols = ["recency", "frequency", "monetary"]
        for col in required_cols:
            assert col in df.columns

    def test_falls_back_to_demo(self, tmp_path, monkeypatch):
        """Test que la fonction utilise les données de démo si pas de fichier."""

        # Simuler un projet sans données
        def mock_root():
            return tmp_path

        monkeypatch.setattr("app.utils.get_project_root", mock_root)

        # Créer la structure de dossiers vide
        (tmp_path / "data" / "processed").mkdir(parents=True)

        df, is_real = load_rfm_data()
        assert is_real is False
        assert len(df) == 5000  # Taille par défaut des données de démo


class TestDemoDataQuality:
    """Tests de qualité des données de démonstration."""

    @pytest.fixture
    def demo_data(self):
        """Génère des données de démo pour les tests."""
        return generate_demo_data(n_samples=5000, random_state=42)

    def test_no_missing_values(self, demo_data):
        """Test qu'il n'y a pas de valeurs manquantes."""
        assert demo_data.isnull().sum().sum() == 0

    def test_no_duplicates_in_index(self, demo_data):
        """Test qu'il n'y a pas de doublons dans l'index."""
        assert not demo_data.index.duplicated().any()

    def test_recency_range_realistic(self, demo_data):
        """Test que la récence a une plage réaliste."""
        assert demo_data["recency"].min() >= 1
        assert demo_data["recency"].max() <= 400

    def test_frequency_range_realistic(self, demo_data):
        """Test que la fréquence a une plage réaliste."""
        assert demo_data["frequency"].min() >= 1
        assert demo_data["frequency"].max() <= 15

    def test_monetary_range_realistic(self, demo_data):
        """Test que le montant a une plage réaliste."""
        assert demo_data["monetary"].min() > 0
        # Les VIP peuvent avoir des montants élevés
        assert demo_data["monetary"].max() < 5000

    def test_vip_have_high_frequency(self, demo_data):
        """Test que les VIP ont une fréquence élevée."""
        vip_freq = demo_data[demo_data["segment"] == 3]["frequency"].mean()
        overall_freq = demo_data["frequency"].mean()
        assert vip_freq > overall_freq

    def test_dormant_have_high_recency(self, demo_data):
        """Test que les dormants ont une récence élevée."""
        dormant_recency = demo_data[demo_data["segment"] == 2]["recency"].mean()
        recent_recency = demo_data[demo_data["segment"] == 0]["recency"].mean()
        assert dormant_recency > recent_recency
