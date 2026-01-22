"""
Tests pour le module src.visualization.plots.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.visualization.plots import (
    plot_elbow_curve,
    plot_silhouette,
    plot_cluster_distribution,
    plot_rfm_boxplots,
    plot_radar_chart,
    plot_scatter_3d,
)


@pytest.fixture
def sample_labels():
    """Génère des labels de test."""
    np.random.seed(42)
    return np.random.randint(0, 4, 100)


@pytest.fixture
def sample_cluster_results():
    """Génère des résultats de clustering de test."""
    return {
        "k_values": [2, 3, 4, 5],
        "inertias": [1000, 800, 600, 500],
        "silhouettes": [0.3, 0.5, 0.6, 0.55],
    }


@pytest.fixture
def sample_scaled_data():
    """Génère des données scalées de test."""
    np.random.seed(42)
    return np.random.randn(100, 3)


class TestPlotElbowCurve:
    """Tests pour plot_elbow_curve."""

    def test_plot_elbow_curve_basic(self, sample_cluster_results):
        """Test de base de plot_elbow_curve."""
        fig = plot_elbow_curve(
            k_values=sample_cluster_results["k_values"],
            inertias=sample_cluster_results["inertias"],
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_elbow_curve_with_silhouette(self, sample_cluster_results):
        """Test avec scores de silhouette."""
        fig = plot_elbow_curve(
            k_values=sample_cluster_results["k_values"],
            inertias=sample_cluster_results["inertias"],
            silhouettes=sample_cluster_results["silhouettes"],
        )

        assert isinstance(fig, plt.Figure)
        # Doit avoir 2 sous-graphiques
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_plot_elbow_curve_save(self, sample_cluster_results, test_data_dir):
        """Test de sauvegarde."""
        save_path = test_data_dir / "elbow.png"
        fig = plot_elbow_curve(
            k_values=sample_cluster_results["k_values"],
            inertias=sample_cluster_results["inertias"],
            save_path=str(save_path),
        )

        assert save_path.exists()
        plt.close(fig)


class TestPlotSilhouette:
    """Tests pour plot_silhouette."""

    def test_plot_silhouette_basic(self, sample_scaled_data, sample_labels):
        """Test de base de plot_silhouette."""
        fig = plot_silhouette(sample_scaled_data, sample_labels)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_silhouette_save(self, sample_scaled_data, sample_labels, test_data_dir):
        """Test de sauvegarde."""
        save_path = test_data_dir / "silhouette.png"
        fig = plot_silhouette(
            sample_scaled_data,
            sample_labels,
            save_path=str(save_path),
        )

        assert save_path.exists()
        plt.close(fig)


class TestPlotClusterDistribution:
    """Tests pour plot_cluster_distribution."""

    def test_plot_cluster_distribution_basic(self, sample_labels):
        """Test de base de plot_cluster_distribution."""
        fig = plot_cluster_distribution(sample_labels)

        assert isinstance(fig, plt.Figure)
        # Doit avoir 2 sous-graphiques (bar + pie)
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_plot_cluster_distribution_save(self, sample_labels, test_data_dir):
        """Test de sauvegarde."""
        save_path = test_data_dir / "distribution.png"
        fig = plot_cluster_distribution(sample_labels, save_path=str(save_path))

        assert save_path.exists()
        plt.close(fig)


class TestPlotRfmBoxplots:
    """Tests pour plot_rfm_boxplots."""

    def test_plot_rfm_boxplots_basic(self, sample_rfm, sample_labels):
        """Test de base de plot_rfm_boxplots."""
        # Ajuster la taille des labels pour correspondre aux données
        labels = sample_labels[: len(sample_rfm)]
        fig = plot_rfm_boxplots(sample_rfm, labels)

        assert isinstance(fig, plt.Figure)
        # Doit avoir 3 sous-graphiques (R, F, M)
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_plot_rfm_boxplots_save(self, sample_rfm, sample_labels, test_data_dir):
        """Test de sauvegarde."""
        labels = sample_labels[: len(sample_rfm)]
        save_path = test_data_dir / "boxplots.png"
        fig = plot_rfm_boxplots(sample_rfm, labels, save_path=str(save_path))

        assert save_path.exists()
        plt.close(fig)


class TestPlotRadarChart:
    """Tests pour plot_radar_chart."""

    def test_plot_radar_chart_basic(self):
        """Test de base de plot_radar_chart."""
        # Créer des centres de clusters de test
        centers = pd.DataFrame(
            {
                "recency": [100, 200, 300, 50],
                "frequency": [5, 2, 1, 8],
                "monetary": [500, 200, 100, 1000],
            },
            index=["Cluster 0", "Cluster 1", "Cluster 2", "Cluster 3"],
        )

        fig = plot_radar_chart(centers)

        # Plotly figure
        assert hasattr(fig, "data")
        assert len(fig.data) == 4  # 4 clusters


class TestPlotScatter3d:
    """Tests pour plot_scatter_3d."""

    def test_plot_scatter_3d_basic(self, sample_rfm, sample_labels):
        """Test de base de plot_scatter_3d."""
        labels = sample_labels[: len(sample_rfm)]
        fig = plot_scatter_3d(sample_rfm, labels)

        # Plotly figure
        assert hasattr(fig, "data")


class TestCleanup:
    """Test de nettoyage des figures."""

    def test_close_all_figures(self):
        """Ferme toutes les figures matplotlib à la fin."""
        plt.close("all")
        assert len(plt.get_fignums()) == 0
