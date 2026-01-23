"""
Module de visualisation pour la segmentation client.

Ce module fournit des fonctions standardisées pour créer
des graphiques de qualité publication.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from src.config import (
    FIGURE_SIZE,
    PLOT_STYLE,
    SEGMENT_COLORS,
    SEGMENT_NAMES,
)

# Configuration globale
sns.set_style(PLOT_STYLE)


def plot_elbow_curve(
    k_values: list[int],
    inertias: list[float],
    silhouettes: list[float] | None = None,
    figsize: tuple = FIGURE_SIZE,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Trace la courbe du coude pour déterminer le nombre optimal de clusters.

    Parameters
    ----------
    k_values : list of int
        Valeurs de k testées.
    inertias : list of float
        Inertie pour chaque k.
    silhouettes : list of float, optional
        Silhouette score pour chaque k.
    figsize : tuple, default (12, 8)
        Taille de la figure.
    save_path : str, optional
        Chemin pour sauvegarder la figure.

    Returns
    -------
    plt.Figure
        Figure matplotlib.
    """
    if silhouettes is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0] // 2, figsize[1]))
        ax2 = None

    # Courbe d'inertie
    ax1.plot(k_values, inertias, "bo-", linewidth=2, markersize=8)
    ax1.set_xlabel("Nombre de clusters (k)", fontsize=12)
    ax1.set_ylabel("Inertie", fontsize=12)
    ax1.set_title("Méthode du coude", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Courbe de silhouette
    if ax2 is not None and silhouettes is not None:
        ax2.plot(k_values, silhouettes, "ro-", linewidth=2, markersize=8)
        ax2.set_xlabel("Nombre de clusters (k)", fontsize=12)
        ax2.set_ylabel("Silhouette Score", fontsize=12)
        ax2.set_title("Score de Silhouette", fontsize=14)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    figsize: tuple = FIGURE_SIZE,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Trace le diagramme de silhouette pour chaque cluster.

    Parameters
    ----------
    X : np.ndarray
        Features scalées.
    labels : np.ndarray
        Labels des clusters.
    figsize : tuple
        Taille de la figure.
    save_path : str, optional
        Chemin pour sauvegarder.

    Returns
    -------
    plt.Figure
        Figure matplotlib.
    """
    from sklearn.metrics import silhouette_samples, silhouette_score

    fig, ax = plt.subplots(figsize=figsize)

    silhouette_avg = silhouette_score(X, labels)
    sample_silhouette_values = silhouette_samples(X, labels)

    n_clusters = len(np.unique(labels))
    y_lower = 10

    for i in range(n_clusters):
        cluster_silhouette_values = sample_silhouette_values[labels == i]
        cluster_silhouette_values.sort()

        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.axvline(
        x=silhouette_avg,
        color="red",
        linestyle="--",
        label=f"Moyenne: {silhouette_avg:.3f}",
    )
    ax.set_xlabel("Coefficient de Silhouette", fontsize=12)
    ax.set_ylabel("Cluster", fontsize=12)
    ax.set_title("Analyse de Silhouette", fontsize=14)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_cluster_distribution(
    labels: np.ndarray,
    figsize: tuple = (10, 6),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Trace la distribution des clients par segment.

    Parameters
    ----------
    labels : np.ndarray
        Labels des clusters.
    figsize : tuple
        Taille de la figure.
    save_path : str, optional
        Chemin pour sauvegarder.

    Returns
    -------
    plt.Figure
        Figure matplotlib.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Comptage
    unique, counts = np.unique(labels, return_counts=True)
    segment_names = [SEGMENT_NAMES.get(i, f"Cluster {i}") for i in unique]
    colors = [SEGMENT_COLORS.get(i, f"C{i}") for i in unique]

    # Bar chart
    ax1.bar(segment_names, counts, color=colors, edgecolor="black")
    ax1.set_xlabel("Segment", fontsize=12)
    ax1.set_ylabel("Nombre de clients", fontsize=12)
    ax1.set_title("Distribution des segments", fontsize=14)

    # Ajouter les valeurs sur les barres
    for i, (_name, count) in enumerate(zip(segment_names, counts, strict=True)):
        ax1.text(i, count + 500, f"{count:,}", ha="center", fontsize=10)

    # Pie chart
    ax2.pie(
        counts,
        labels=segment_names,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        explode=[0.02] * len(unique),
    )
    ax2.set_title("Répartition des segments", fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_rfm_boxplots(
    df: pd.DataFrame,
    labels: np.ndarray,
    figsize: tuple = (15, 5),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Trace les boxplots RFM par segment.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame avec les colonnes recency, frequency, monetary.
    labels : np.ndarray
        Labels des clusters.
    figsize : tuple
        Taille de la figure.
    save_path : str, optional
        Chemin pour sauvegarder.

    Returns
    -------
    plt.Figure
        Figure matplotlib.
    """
    df = df.copy()
    df["segment"] = [SEGMENT_NAMES.get(label, f"Cluster {label}") for label in labels]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    features = ["recency", "frequency", "monetary"]
    titles = ["Récence (jours)", "Fréquence", "Montant (BRL)"]

    # Créer une palette avec les noms de segments comme clés
    palette = {
        SEGMENT_NAMES.get(k, f"Cluster {k}"): v for k, v in SEGMENT_COLORS.items()
    }

    for ax, feature, title in zip(axes, features, titles, strict=True):
        sns.boxplot(
            data=df,
            x="segment",
            y=feature,
            ax=ax,
            hue="segment",
            palette=palette,
            legend=False,
        )
        ax.set_xlabel("Segment", fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f"Distribution de {title}", fontsize=12)
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_radar_chart(
    cluster_centers: pd.DataFrame,
    save_path: str | None = None,
) -> go.Figure:
    """
    Trace un radar chart des profils de segments.

    Parameters
    ----------
    cluster_centers : pd.DataFrame
        Centres des clusters (normalisés de préférence).
    save_path : str, optional
        Chemin pour sauvegarder.

    Returns
    -------
    plotly.graph_objects.Figure
        Figure Plotly interactive.
    """
    from sklearn.preprocessing import MinMaxScaler

    # Normalisation pour le radar
    scaler = MinMaxScaler()
    centers_normalized = pd.DataFrame(
        scaler.fit_transform(cluster_centers),
        columns=cluster_centers.columns,
        index=cluster_centers.index,
    )

    fig = go.Figure()

    categories = centers_normalized.columns.tolist()
    categories += [categories[0]]  # Fermer le polygone

    for i, (_idx, row) in enumerate(centers_normalized.iterrows()):
        values = row.tolist()
        values += [values[0]]  # Fermer le polygone

        segment_name = SEGMENT_NAMES.get(i, f"Cluster {i}")

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories,
                fill="toself",
                name=segment_name,
                line_color=SEGMENT_COLORS.get(
                    i, f"rgb({i*60}, {100+i*30}, {200-i*40})"
                ),
            )
        )

    fig.update_layout(
        polar={
            "radialaxis": {"visible": True, "range": [0, 1]},
        },
        showlegend=True,
        title="Profil des segments (normalisé)",
    )

    if save_path:
        fig.write_image(save_path)

    return fig


def plot_scatter_3d(
    df: pd.DataFrame,
    labels: np.ndarray,
    save_path: str | None = None,
) -> go.Figure:
    """
    Trace un scatter plot 3D des clusters.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame avec recency, frequency, monetary.
    labels : np.ndarray
        Labels des clusters.
    save_path : str, optional
        Chemin pour sauvegarder.

    Returns
    -------
    plotly.graph_objects.Figure
        Figure Plotly interactive.
    """
    df = df.copy()
    df["segment"] = [SEGMENT_NAMES.get(label, f"Cluster {label}") for label in labels]

    fig = px.scatter_3d(
        df,
        x="recency",
        y="frequency",
        z="monetary",
        color="segment",
        color_discrete_map={SEGMENT_NAMES[k]: v for k, v in SEGMENT_COLORS.items()},
        title="Visualisation 3D des segments",
        labels={
            "recency": "Récence (jours)",
            "frequency": "Fréquence",
            "monetary": "Montant (BRL)",
        },
        opacity=0.7,
    )

    fig.update_layout(
        scene={
            "xaxis_title": "Récence",
            "yaxis_title": "Fréquence",
            "zaxis_title": "Montant",
        }
    )

    if save_path:
        fig.write_html(save_path)

    return fig
