"""
Evaluate customer segmentation model.

This script:
1. Loads a trained model
2. Loads clustered data
3. Computes evaluation metrics
4. Generates evaluation visualizations
"""

import argparse
import yaml
import logging
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.models.clustering import CustomerSegmenter
from src.data.data_loader import load_processed_data
from src.evaluation.metrics import evaluate_clustering, ClusteringEvaluator
from src.visualization.plots import (
    plot_clusters_2d,
    plot_cluster_profiles,
    plot_silhouette_analysis
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    """Main evaluation pipeline."""
    logger.info("=" * 50)
    logger.info("CUSTOMER SEGMENTATION - EVALUATION PIPELINE")
    logger.info("=" * 50)

    # Load configuration
    config = load_config(args.config)
    logger.info(f"Configuration loaded from {args.config}")

    # Load trained model
    logger.info("\n[1/4] Loading trained model...")
    model_path = Path(config['model']['save_dir']) / config['model']['model_filename']

    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        logger.error("Please train a model first using train.py")
        return

    segmenter = CustomerSegmenter.load_model(str(model_path))
    logger.info(f"Model loaded from {model_path}")
    logger.info(f"Algorithm: {segmenter.algorithm}")
    logger.info(f"Features: {segmenter.feature_columns}")

    # Load clustered data
    logger.info("\n[2/4] Loading clustered data...")
    data_path = Path(config['data']['processed_dir']) / "rfm_clustered.csv"

    if not data_path.exists():
        logger.error(f"Clustered data not found at {data_path}")
        logger.error("Please run train.py first to generate clustered data")
        return

    data = load_processed_data(str(data_path))
    logger.info(f"Loaded {len(data)} customers")

    # Evaluate clustering
    logger.info("\n[3/4] Computing evaluation metrics...")
    feature_columns = segmenter.feature_columns
    labels = data['Cluster'].values

    # Compute metrics
    metrics = evaluate_clustering(data, labels, feature_columns)

    # Save metrics
    viz_dir = Path(config['visualization']['save_dir'])
    viz_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = viz_dir / "evaluation_report.yaml"
    with open(metrics_path, 'w') as f:
        yaml.dump(metrics, f)
    logger.info(f"Metrics saved to {metrics_path}")

    # Generate evaluation visualizations
    logger.info("\n[4/4] Generating evaluation visualizations...")

    # Cluster profiles
    logger.info("Creating cluster profiles...")
    plot_cluster_profiles(
        data,
        save_path=viz_dir / "cluster_profiles_eval.png"
    )

    # 2D scatter plots for different feature pairs
    logger.info("Creating 2D cluster visualizations...")
    feature_pairs = [
        ('Recency', 'Monetary'),
        ('Frequency', 'Monetary'),
        ('Recency', 'Frequency')
    ]

    for x_col, y_col in feature_pairs:
        if x_col in data.columns and y_col in data.columns:
            plot_clusters_2d(
                data,
                x_col=x_col,
                y_col=y_col,
                save_path=viz_dir / f"clusters_{x_col.lower()}_{y_col.lower()}_eval.png"
            )

    # Silhouette analysis
    logger.info("Creating silhouette analysis...")
    scaler = StandardScaler()
    X = data[feature_columns].values
    X_scaled = scaler.fit_transform(X)

    plot_silhouette_analysis(
        X_scaled,
        labels,
        save_path=viz_dir / "silhouette_analysis.png"
    )

    # Print detailed cluster statistics
    logger.info("\n" + "=" * 50)
    logger.info("DETAILED CLUSTER STATISTICS")
    logger.info("=" * 50)

    for cluster in sorted(data['Cluster'].unique()):
        cluster_data = data[data['Cluster'] == cluster]
        cluster_desc = config['cluster_descriptions'].get(cluster, f"Cluster {cluster}")

        logger.info(f"\n--- {cluster_desc} (Cluster {cluster}) ---")
        logger.info(f"Size: {len(cluster_data)} customers ({len(cluster_data)/len(data)*100:.1f}%)")
        logger.info(f"\nRFM Statistics:")

        for feature in feature_columns:
            mean_val = cluster_data[feature].mean()
            median_val = cluster_data[feature].median()
            std_val = cluster_data[feature].std()
            logger.info(f"  {feature}: mean={mean_val:.2f}, median={median_val:.2f}, std={std_val:.2f}")

    # Print evaluation summary
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Number of clusters: {metrics['n_clusters']}")
    logger.info(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
    logger.info(f"Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.2f}")
    logger.info(f"Davies-Bouldin Score: {metrics['davies_bouldin_score']:.4f}")

    logger.info("\nInterpretation:")
    logger.info("- Silhouette Score: [-1, 1], higher is better (>0.5 is good)")
    logger.info("- Calinski-Harabasz: Higher is better (well-separated clusters)")
    logger.info("- Davies-Bouldin: Lower is better (<1 is good)")

    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION COMPLETE!")
    logger.info(f"Report saved to: {metrics_path}")
    logger.info(f"Visualizations saved to: {viz_dir}")
    logger.info("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate customer segmentation model")
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )

    args = parser.parse_args()
    main(args)
