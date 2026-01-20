"""
Train customer segmentation model.

This script:
1. Loads RFM data
2. Trains clustering model
3. Evaluates model performance
4. Saves trained model and visualizations
"""

import argparse
import yaml
import logging
from pathlib import Path
import pandas as pd

from src.data.data_loader import load_processed_data
from src.models.clustering import CustomerSegmenter, find_optimal_k
from src.evaluation.metrics import evaluate_clustering
from src.visualization.plots import (
    plot_rfm_distributions,
    plot_clusters_2d,
    plot_clusters_3d_interactive,
    plot_elbow_curve,
    plot_cluster_profiles
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
    """Main training pipeline."""
    logger.info("=" * 50)
    logger.info("CUSTOMER SEGMENTATION - TRAINING PIPELINE")
    logger.info("=" * 50)

    # Load configuration
    config = load_config(args.config)
    logger.info(f"Configuration loaded from {args.config}")

    # Create output directories
    model_dir = Path(config['model']['save_dir'])
    viz_dir = Path(config['visualization']['save_dir'])
    model_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Load RFM data
    logger.info("\n[1/6] Loading RFM data...")
    data_path = f"{config['data']['processed_dir']}/{config['data']['processed_files']['rfm']}"
    rfm_data = load_processed_data(data_path)
    logger.info(f"Loaded {len(rfm_data)} customers")

    # Plot RFM distributions
    logger.info("\n[2/6] Visualizing RFM distributions...")
    plot_rfm_distributions(
        rfm_data,
        save_path=viz_dir / "rfm_distributions.png"
    )

    # Find optimal K (if using K-Means and elbow method enabled)
    feature_columns = config['clustering']['feature_columns']
    if config['clustering']['algorithm'] == 'kmeans' and args.find_optimal_k:
        logger.info("\n[3/6] Finding optimal number of clusters (Elbow Method)...")
        k_range = range(
            config['clustering']['elbow']['k_min'],
            config['clustering']['elbow']['k_max'] + 1
        )
        k_values, inertias = find_optimal_k(
            rfm_data,
            feature_columns=feature_columns,
            k_range=k_range,
            random_state=config['clustering']['kmeans']['random_state']
        )
        plot_elbow_curve(
            k_values,
            inertias,
            save_path=viz_dir / "elbow_curve.png"
        )
    else:
        logger.info("\n[3/6] Skipping optimal K search (using configured value)...")

    # Train clustering model
    logger.info("\n[4/6] Training clustering model...")
    algorithm = config['clustering']['algorithm']
    algo_config = config['clustering'][algorithm]

    segmenter = CustomerSegmenter(algorithm=algorithm, **algo_config)
    labels = segmenter.fit_predict(rfm_data, feature_columns=feature_columns)

    # Add cluster labels to data
    rfm_data['Cluster'] = labels

    # Evaluate model
    logger.info("\n[5/6] Evaluating model performance...")
    metrics = evaluate_clustering(rfm_data, labels, feature_columns)

    # Save metrics to file
    metrics_path = model_dir / "evaluation_metrics.yaml"
    with open(metrics_path, 'w') as f:
        yaml.dump(metrics, f)
    logger.info(f"Metrics saved to {metrics_path}")

    # Generate visualizations
    logger.info("\n[6/6] Generating visualizations...")

    # 2D scatter plots
    plot_clusters_2d(
        rfm_data,
        x_col='Recency',
        y_col='Monetary',
        save_path=viz_dir / "clusters_recency_monetary.png"
    )

    plot_clusters_2d(
        rfm_data,
        x_col='Frequency',
        y_col='Monetary',
        save_path=viz_dir / "clusters_frequency_monetary.png"
    )

    # 3D interactive plot
    plot_clusters_3d_interactive(
        rfm_data,
        save_path=viz_dir / "clusters_3d_interactive.html"
    )

    # Cluster profiles
    plot_cluster_profiles(
        rfm_data,
        save_path=viz_dir / "cluster_profiles.png"
    )

    # Save trained model
    model_path = model_dir / config['model']['model_filename']
    segmenter.save_model(str(model_path))

    # Save clustered data
    output_data_path = Path(config['data']['processed_dir']) / "rfm_clustered.csv"
    rfm_data.to_csv(output_data_path, index=False)
    logger.info(f"Clustered data saved to {output_data_path}")

    # Print cluster summary
    logger.info("\n" + "=" * 50)
    logger.info("CLUSTER SUMMARY")
    logger.info("=" * 50)
    cluster_summary = rfm_data.groupby('Cluster')[feature_columns].agg(['mean', 'count'])
    logger.info(f"\n{cluster_summary}")

    logger.info("\n" + "=" * 50)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Visualizations saved to: {viz_dir}")
    logger.info("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train customer segmentation model")
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--find-optimal-k',
        action='store_true',
        help='Run elbow method to find optimal K (for K-Means only)'
    )

    args = parser.parse_args()
    main(args)
