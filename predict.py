"""
Predict cluster assignments for new customers.

This script:
1. Loads a trained segmentation model
2. Loads new customer RFM data
3. Predicts cluster assignments
4. Saves results with cluster labels
"""

import argparse
import yaml
import logging
from pathlib import Path
import pandas as pd

from src.models.clustering import CustomerSegmenter
from src.data.data_loader import load_processed_data, save_processed_data

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
    """Main prediction pipeline."""
    logger.info("=" * 50)
    logger.info("CUSTOMER SEGMENTATION - PREDICTION PIPELINE")
    logger.info("=" * 50)

    # Load configuration
    config = load_config(args.config)
    logger.info(f"Configuration loaded from {args.config}")

    # Load trained model
    logger.info("\n[1/3] Loading trained model...")
    model_path = Path(config['model']['save_dir']) / config['model']['model_filename']

    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        logger.error("Please train a model first using train.py")
        return

    segmenter = CustomerSegmenter.load_model(str(model_path))
    logger.info(f"Model loaded from {model_path}")
    logger.info(f"Algorithm: {segmenter.algorithm}")
    logger.info(f"Features: {segmenter.feature_columns}")

    # Load new customer data
    logger.info("\n[2/3] Loading customer data...")
    if args.input:
        input_path = args.input
    else:
        # Use default RFM data path
        input_path = f"{config['data']['processed_dir']}/{config['data']['processed_files']['rfm']}"

    new_data = load_processed_data(input_path)
    logger.info(f"Loaded {len(new_data)} customers from {input_path}")

    # Predict cluster assignments
    logger.info("\n[3/3] Predicting cluster assignments...")

    if segmenter.algorithm == 'kmeans':
        # K-Means supports predict
        labels = segmenter.predict(new_data)
    else:
        # For DBSCAN and Agglomerative, we need to re-fit (limitation)
        logger.warning(f"{segmenter.algorithm} doesn't support prediction on new data.")
        logger.warning("Re-fitting model on the provided data...")
        labels = segmenter.fit_predict(new_data, feature_columns=segmenter.feature_columns)

    # Add cluster labels
    new_data['Cluster'] = labels

    # Get cluster descriptions from config
    cluster_descriptions = config.get('cluster_descriptions', {})
    new_data['Cluster_Description'] = new_data['Cluster'].map(
        lambda x: cluster_descriptions.get(x, f"Cluster {x}")
    )

    # Save results
    if args.output:
        output_path = args.output
    else:
        output_path = Path(config['data']['processed_dir']) / "predictions.csv"

    save_processed_data(new_data, str(output_path))
    logger.info(f"Predictions saved to {output_path}")

    # Print cluster distribution
    logger.info("\n" + "=" * 50)
    logger.info("CLUSTER DISTRIBUTION")
    logger.info("=" * 50)
    cluster_counts = new_data['Cluster'].value_counts().sort_index()
    for cluster, count in cluster_counts.items():
        pct = (count / len(new_data)) * 100
        desc = cluster_descriptions.get(cluster, f"Cluster {cluster}")
        logger.info(f"Cluster {cluster} ({desc}): {count} customers ({pct:.1f}%)")

    logger.info("\n" + "=" * 50)
    logger.info("PREDICTION COMPLETE!")
    logger.info(f"Results saved to: {output_path}")
    logger.info("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict customer segments")
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Path to input CSV file with RFM data (default: from config)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save predictions (default: data/processed/predictions.csv)'
    )

    args = parser.parse_args()
    main(args)
