"""
Tests pour le module de monitoring MLOps.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.monitoring.drift import (
    DriftDetector,
    DriftReport,
    calculate_ari,
    calculate_data_drift,
    check_model_drift,
    estimate_retraining_frequency,
)
from src.monitoring.registry import ModelRegistry

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_rfm_data():
    """Génère des données RFM de test."""
    np.random.seed(42)
    n_samples = 500

    return pd.DataFrame(
        {
            "recency": np.random.exponential(100, n_samples),
            "frequency": np.random.poisson(3, n_samples) + 1,
            "monetary": np.random.lognormal(4, 1, n_samples),
        }
    )


@pytest.fixture
def trained_model(sample_rfm_data):
    """Retourne un modèle KMeans entraîné."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sample_rfm_data)

    model = KMeans(n_clusters=4, random_state=42, n_init=10)
    model.fit(X_scaled)

    return model, scaler


@pytest.fixture
def temp_registry_dir():
    """Crée un dossier temporaire pour le registre."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# Tests ModelRegistry
# ============================================================================


class TestModelRegistry:
    """Tests pour le Model Registry."""

    def test_init_creates_directory(self, temp_registry_dir):
        """Test que l'initialisation crée le dossier."""
        _registry = ModelRegistry(temp_registry_dir)  # noqa: F841
        assert temp_registry_dir.exists()
        assert (temp_registry_dir / "registry.json").exists()

    def test_compute_data_hash(self, sample_rfm_data):
        """Test le calcul du hash des données."""
        hash1 = ModelRegistry.compute_data_hash(sample_rfm_data)
        hash2 = ModelRegistry.compute_data_hash(sample_rfm_data)

        assert isinstance(hash1, str)
        assert len(hash1) == 16
        assert hash1 == hash2  # Reproductible

    def test_compute_data_hash_different_data(self, sample_rfm_data):
        """Test que des données différentes ont des hashs différents."""
        hash1 = ModelRegistry.compute_data_hash(sample_rfm_data)

        modified_data = sample_rfm_data.copy()
        modified_data.iloc[0, 0] = 999999
        hash2 = ModelRegistry.compute_data_hash(modified_data)

        assert hash1 != hash2

    def test_register_model(self, temp_registry_dir, trained_model):
        """Test l'enregistrement d'un modèle."""
        model, scaler = trained_model
        registry = ModelRegistry(temp_registry_dir)

        version = registry.register(
            model=model,
            scaler=scaler,
            metrics={"silhouette": 0.65},
            hyperparameters={"n_clusters": 4},
            description="Test model",
        )

        assert version == "0.0.1"
        assert (temp_registry_dir / "v0.0.1" / "model.pkl").exists()
        assert (temp_registry_dir / "v0.0.1" / "scaler.pkl").exists()
        assert (temp_registry_dir / "v0.0.1" / "metadata.json").exists()

    def test_register_increments_version(self, temp_registry_dir, trained_model):
        """Test que les versions s'incrémentent."""
        model, scaler = trained_model
        registry = ModelRegistry(temp_registry_dir)

        v1 = registry.register(model=model, description="v1")
        v2 = registry.register(model=model, description="v2")
        v3 = registry.register(model=model, description="v3", bump="minor")

        assert v1 == "0.0.1"
        assert v2 == "0.0.2"
        assert v3 == "0.1.0"

    def test_load_model(self, temp_registry_dir, trained_model):
        """Test le chargement d'un modèle."""
        model, scaler = trained_model
        registry = ModelRegistry(temp_registry_dir)

        registry.register(
            model=model,
            scaler=scaler,
            metrics={"silhouette": 0.65},
        )

        loaded_model, loaded_scaler, metadata = registry.load()

        assert loaded_model is not None
        assert loaded_scaler is not None
        assert metadata["version"] == "0.0.1"
        assert metadata["metrics"]["silhouette"] == 0.65

    def test_load_specific_version(self, temp_registry_dir, trained_model):
        """Test le chargement d'une version spécifique."""
        model, scaler = trained_model
        registry = ModelRegistry(temp_registry_dir)

        registry.register(model=model, description="v1")
        registry.register(model=model, description="v2")

        _, _, meta1 = registry.load("0.0.1")
        _, _, meta2 = registry.load("0.0.2")

        assert meta1["description"] == "v1"
        assert meta2["description"] == "v2"

    def test_list_versions(self, temp_registry_dir, trained_model):
        """Test la liste des versions."""
        model, scaler = trained_model
        registry = ModelRegistry(temp_registry_dir)

        registry.register(model=model, metrics={"silhouette": 0.65})
        registry.register(model=model, metrics={"silhouette": 0.70})

        versions = registry.list_versions()

        assert len(versions) == 2
        assert "version" in versions.columns
        assert "metric_silhouette" in versions.columns

    def test_compare_versions(self, temp_registry_dir, trained_model):
        """Test la comparaison de versions."""
        model, scaler = trained_model
        registry = ModelRegistry(temp_registry_dir)

        registry.register(
            model=model,
            metrics={"silhouette": 0.65},
            hyperparameters={"n_clusters": 4},
        )
        registry.register(
            model=model,
            metrics={"silhouette": 0.70},
            hyperparameters={"n_clusters": 5},
        )

        comparison = registry.compare_versions("0.0.1", "0.0.2")

        assert len(comparison) > 0
        assert "v0.0.1" in comparison.columns
        assert "v0.0.2" in comparison.columns

    def test_delete_version(self, temp_registry_dir, trained_model):
        """Test la suppression d'une version."""
        model, scaler = trained_model
        registry = ModelRegistry(temp_registry_dir)

        registry.register(model=model)
        registry.register(model=model)

        result = registry.delete_version("0.0.1")

        assert result is True
        assert not (temp_registry_dir / "v0.0.1").exists()
        assert len(registry.list_versions()) == 1


# ============================================================================
# Tests Drift Detection
# ============================================================================


class TestDriftFunctions:
    """Tests pour les fonctions de détection de drift."""

    def test_calculate_ari_identical(self):
        """Test ARI avec labels identiques."""
        labels = np.array([0, 0, 1, 1, 2, 2])
        ari = calculate_ari(labels, labels)
        assert ari == 1.0

    def test_calculate_ari_different(self):
        """Test ARI avec labels différents."""
        labels1 = np.array([0, 0, 1, 1, 2, 2])
        labels2 = np.array([1, 1, 0, 0, 2, 2])

        ari = calculate_ari(labels1, labels2)
        assert -1 <= ari <= 1

    def test_calculate_data_drift_no_drift(self, sample_rfm_data):
        """Test détection sans drift."""
        drift = calculate_data_drift(sample_rfm_data, sample_rfm_data)

        for _feature, result in drift.items():
            assert result["drift_detected"] == False  # noqa: E712
            assert result["pvalue"] > 0.05

    def test_calculate_data_drift_with_drift(self, sample_rfm_data):
        """Test détection avec drift."""
        shifted_data = sample_rfm_data.copy()
        shifted_data["recency"] = shifted_data["recency"] + 100  # Shift important

        drift = calculate_data_drift(sample_rfm_data, shifted_data)

        assert drift["recency"]["drift_detected"] == True  # noqa: E712
        assert drift["recency"]["pvalue"] < 0.05

    def test_check_model_drift(self, sample_rfm_data, trained_model):
        """Test la vérification du drift de modèle."""
        model, scaler = trained_model

        # Sans drift
        drift_detected, ari = check_model_drift(
            model, scaler, sample_rfm_data, sample_rfm_data
        )

        assert isinstance(drift_detected, bool)
        assert 0 <= ari <= 1


class TestDriftDetector:
    """Tests pour la classe DriftDetector."""

    def test_init(self):
        """Test l'initialisation."""
        detector = DriftDetector(ari_threshold=0.8, pvalue_threshold=0.05)
        assert detector.ari_threshold == 0.8
        assert detector.pvalue_threshold == 0.05
        assert detector._is_fitted is False

    def test_fit(self, sample_rfm_data):
        """Test la configuration du détecteur."""
        labels = np.random.randint(0, 4, len(sample_rfm_data))
        detector = DriftDetector()
        detector.fit(sample_rfm_data, labels)

        assert detector._is_fitted is True
        assert detector.reference_data is not None
        assert detector.reference_labels is not None

    def test_detect_without_fit_raises(self, sample_rfm_data, trained_model):
        """Test que detect sans fit lève une erreur."""
        model, scaler = trained_model
        detector = DriftDetector()

        with pytest.raises(ValueError, match="configuré"):
            detector.detect(sample_rfm_data, model, scaler)

    def test_detect_returns_report(self, sample_rfm_data, trained_model):
        """Test que detect retourne un DriftReport."""
        model, scaler = trained_model
        labels = model.predict(scaler.transform(sample_rfm_data))

        detector = DriftDetector()
        detector.fit(sample_rfm_data, labels)

        report = detector.detect(sample_rfm_data, model, scaler)

        assert isinstance(report, DriftReport)
        assert isinstance(report.timestamp, datetime)
        assert isinstance(report.data_drift_detected, bool)
        assert isinstance(report.model_drift_detected, bool)
        assert report.recommendation != ""

    def test_simulate_temporal_drift(self, sample_rfm_data, trained_model):
        """Test la simulation de drift temporel."""
        model, scaler = trained_model

        detector = DriftDetector()
        simulation = detector.simulate_temporal_drift(
            data=sample_rfm_data,
            model=model,
            scaler=scaler,
            n_periods=6,
            drift_rate=0.05,
        )

        assert len(simulation) == 6
        assert "period" in simulation.columns
        assert "ari_score" in simulation.columns
        assert "needs_retraining" in simulation.columns


class TestEstimateRetrainingFrequency:
    """Tests pour l'estimation de fréquence de réentraînement."""

    def test_returns_dict(self, sample_rfm_data, trained_model):
        """Test que la fonction retourne un dict."""
        model, scaler = trained_model

        result = estimate_retraining_frequency(
            data=sample_rfm_data,
            model=model,
            scaler=scaler,
            max_periods=6,
        )

        assert isinstance(result, dict)
        assert "periods_until_drift" in result
        assert "ari_at_drift" in result
        assert "recommendation" in result
        assert "simulation_data" in result

    def test_periods_is_positive(self, sample_rfm_data, trained_model):
        """Test que le nombre de périodes est positif."""
        model, scaler = trained_model

        result = estimate_retraining_frequency(
            data=sample_rfm_data,
            model=model,
            scaler=scaler,
        )

        assert result["periods_until_drift"] > 0


class TestDriftReport:
    """Tests pour DriftReport."""

    def test_to_dict(self):
        """Test la conversion en dictionnaire."""
        report = DriftReport(
            timestamp=datetime.now(),
            data_drift_detected=True,
            model_drift_detected=False,
            ari_score=0.85,
            feature_drifts={"recency": {"drift_detected": True}},
            recommendation="Test recommendation",
        )

        d = report.to_dict()

        assert isinstance(d, dict)
        assert "timestamp" in d
        assert d["data_drift_detected"] is True
        assert d["model_drift_detected"] is False
        assert d["ari_score"] == 0.85
