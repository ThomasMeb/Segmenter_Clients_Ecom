"""
Visualization utilities for customer segmentation analysis.

This module provides plotting functions for:
- Cluster visualization (2D/3D scatter plots)
- RFM distributions
- Elbow curves
- Silhouette plots
- Cluster profiles
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_rfm_distributions(
    data: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Plot distributions of Recency, Frequency, and Monetary values.

    Parameters:
        data (pd.DataFrame): RFM data.
        save_path (str, optional): Path to save the figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Recency
    axes[0].hist(data['Recency'], bins=50, color='skyblue', edgecolor='black')
    axes[0].set_title('Recency Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Days Since Last Purchase')
    axes[0].set_ylabel('Number of Customers')

    # Frequency
    axes[1].hist(data['Frequency'], bins=30, color='lightcoral', edgecolor='black')
    axes[1].set_title('Frequency Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Number of Purchases')
    axes[1].set_ylabel('Number of Customers')

    # Monetary
    axes[2].hist(data['Monetary'], bins=50, color='lightgreen', edgecolor='black')
    axes[2].set_title('Monetary Distribution', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Total Spend ($)')
    axes[2].set_ylabel('Number of Customers')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"RFM distribution plot saved to {save_path}")

    plt.show()


def plot_clusters_2d(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    cluster_col: str = 'Cluster',
    save_path: Optional[str] = None
) -> None:
    """
    Plot 2D scatter plot of clusters.

    Parameters:
        data (pd.DataFrame): Data with cluster labels.
        x_col (str): Column name for x-axis.
        y_col (str): Column name for y-axis.
        cluster_col (str): Column name for cluster labels.
        save_path (str, optional): Path to save the figure.
    """
    plt.figure(figsize=(12, 8))

    # Create scatter plot
    clusters = data[cluster_col].unique()
    colors = sns.color_palette("husl", len(clusters))

    for i, cluster in enumerate(sorted(clusters)):
        if cluster == -1:  # Noise points for DBSCAN
            color = 'gray'
            label = 'Noise'
            alpha = 0.3
        else:
            color = colors[i]
            label = f'Cluster {cluster}'
            alpha = 0.6

        cluster_data = data[data[cluster_col] == cluster]
        plt.scatter(
            cluster_data[x_col],
            cluster_data[y_col],
            c=[color],
            label=label,
            alpha=alpha,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )

    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.title(f'{x_col} vs {y_col} by Cluster', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"2D cluster plot saved to {save_path}")

    plt.show()


def plot_clusters_3d_interactive(
    data: pd.DataFrame,
    cluster_col: str = 'Cluster',
    save_path: Optional[str] = None
) -> None:
    """
    Create interactive 3D scatter plot of RFM clusters.

    Parameters:
        data (pd.DataFrame): Data with RFM features and cluster labels.
        cluster_col (str): Column name for cluster labels.
        save_path (str, optional): Path to save the HTML figure.
    """
    fig = px.scatter_3d(
        data,
        x='Recency',
        y='Frequency',
        z='Monetary',
        color=cluster_col,
        title='3D Customer Segmentation (RFM)',
        labels={'Recency': 'Recency (days)', 'Frequency': 'Frequency', 'Monetary': 'Monetary ($)'},
        opacity=0.7,
        color_continuous_scale='viridis' if data[cluster_col].dtype != 'object' else None
    )

    fig.update_traces(marker=dict(size=4))

    if save_path:
        fig.write_html(save_path)
        logger.info(f"3D interactive plot saved to {save_path}")

    fig.show()


def plot_elbow_curve(
    k_values: List[int],
    inertias: List[float],
    save_path: Optional[str] = None
) -> None:
    """
    Plot elbow curve for K-Means clustering.

    Parameters:
        k_values (List[int]): List of k values tested.
        inertias (List[float]): Corresponding inertia values.
        save_path (str, optional): Path to save the figure.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertias, marker='o', linewidth=2, markersize=8, color='steelblue')
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
    plt.title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Elbow curve saved to {save_path}")

    plt.show()


def plot_cluster_profiles(
    data: pd.DataFrame,
    cluster_col: str = 'Cluster',
    save_path: Optional[str] = None
) -> None:
    """
    Plot cluster profiles showing mean RFM values.

    Parameters:
        data (pd.DataFrame): Data with cluster labels.
        cluster_col (str): Column name for cluster labels.
        save_path (str, optional): Path to save the figure.
    """
    # Calculate mean values per cluster
    cluster_means = data.groupby(cluster_col)[['Recency', 'Frequency', 'Monetary']].mean()

    # Normalize for radar chart
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    cluster_means_scaled = pd.DataFrame(
        scaler.fit_transform(cluster_means),
        columns=cluster_means.columns,
        index=cluster_means.index
    )

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Bar plot
    cluster_means.plot(kind='bar', ax=axes[0], color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[0].set_title('Cluster Profiles (Mean RFM Values)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Cluster')
    axes[0].set_ylabel('Mean Value')
    axes[0].legend(title='Metric')
    axes[0].grid(True, alpha=0.3)

    # Heatmap
    sns.heatmap(cluster_means.T, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[1], cbar_kws={'label': 'Mean Value'})
    axes[1].set_title('Cluster Profiles Heatmap', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Cluster')
    axes[1].set_ylabel('Metric')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Cluster profiles plot saved to {save_path}")

    plt.show()


def plot_silhouette_analysis(
    X: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plot silhouette analysis for clustering.

    Parameters:
        X (np.ndarray): Feature matrix (scaled).
        labels (np.ndarray): Cluster labels.
        save_path (str, optional): Path to save the figure.
    """
    from sklearn.metrics import silhouette_samples, silhouette_score

    n_clusters = len(np.unique(labels[labels != -1]))
    silhouette_avg = silhouette_score(X, labels)

    fig, ax = plt.subplots(figsize=(10, 6))

    y_lower = 10
    for i in range(n_clusters):
        # Get silhouette scores for cluster i
        cluster_silhouette_values = silhouette_samples(X, labels)
        ith_cluster_silhouette_values = cluster_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7
        )

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
    ax.set_xlabel('Silhouette Coefficient')
    ax.set_ylabel('Cluster')
    ax.axvline(x=silhouette_avg, color="red", linestyle="--", label=f'Average: {silhouette_avg:.3f}')
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Silhouette analysis plot saved to {save_path}")

    plt.show()


def plot_temporal_stability(
    stability_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Plot temporal stability analysis (ARI over time).

    Parameters:
        stability_df (pd.DataFrame): DataFrame with week and ARI columns.
        save_path (str, optional): Path to save the figure.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(stability_df['week'], stability_df['ari'], marker='o', linewidth=2, markersize=6, color='steelblue')
    plt.xlabel('Weeks Since Base Clustering', fontsize=12)
    plt.ylabel('Adjusted Rand Index (ARI)', fontsize=12)
    plt.title('Clustering Temporal Stability', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.8, color='red', linestyle='--', label='Threshold (0.8)')
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Temporal stability plot saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    # Example usage
    from src.data.data_loader import load_processed_data
    from src.models.clustering import CustomerSegmenter

    # Load data
    rfm_data = load_processed_data("data/processed/data_RFM.csv")

    # Train model
    segmenter = CustomerSegmenter(algorithm='kmeans', n_clusters=4, random_state=42)
    labels = segmenter.fit_predict(rfm_data, feature_columns=['Recency', 'Frequency', 'Monetary'])
    rfm_data['Cluster'] = labels

    # Create visualizations
    plot_rfm_distributions(rfm_data, save_path='outputs/figures/rfm_distributions.png')
    plot_clusters_2d(rfm_data, 'Recency', 'Monetary', save_path='outputs/figures/clusters_2d.png')
    plot_cluster_profiles(rfm_data, save_path='outputs/figures/cluster_profiles.png')
