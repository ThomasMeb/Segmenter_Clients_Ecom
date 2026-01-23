"""
CLI for Olist Customer Segmentation.

This module provides command-line interface for training, prediction,
evaluation and serving the customer segmentation model.

Usage:
    olist-segment train --input data/raw/data.csv --output models/
    olist-segment predict --model models/ --input new_customers.csv
    olist-segment evaluate --model models/ --data data/processed/customers_rfm.parquet
    olist-segment serve --port 8501
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import pandas as pd

from src.config import (
    MODELS_DIR,
    N_CLUSTERS,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RAW_DATA_DIR,
)


@click.group()
@click.version_option(version="1.0.0", prog_name="olist-segment")
def main() -> None:
    """Olist Customer Segmentation CLI.

    A tool for RFM-based customer segmentation using KMeans clustering.
    """
    pass


@main.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to input CSV file with transactions.",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to save model and processed data.",
)
@click.option(
    "--n-clusters",
    "-k",
    default=N_CLUSTERS,
    type=int,
    help=f"Number of clusters (default: {N_CLUSTERS}).",
)
@click.option(
    "--random-state",
    default=RANDOM_STATE,
    type=int,
    help=f"Random state for reproducibility (default: {RANDOM_STATE}).",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output.")
def train(
    input_path: Path | None,
    output_path: Path | None,
    n_clusters: int,
    random_state: int,
    verbose: bool,
) -> None:
    """Train the customer segmentation model.

    This command:
    1. Loads and cleans transaction data
    2. Calculates RFM features
    3. Trains a KMeans clustering model
    4. Saves the model and processed data
    """
    from src.data.preprocessor import clean_transactions
    from src.features.rfm import RFMCalculator
    from src.models.clustering import CustomerSegmenter
    from src.models.evaluation import evaluate_clustering

    # Default paths
    if input_path is None:
        input_path = RAW_DATA_DIR / "data.csv"
    if output_path is None:
        output_path = MODELS_DIR

    click.echo(click.style("=" * 60, fg="blue"))
    click.echo(
        click.style("  Olist Customer Segmentation - Training", fg="blue", bold=True)
    )
    click.echo(click.style("=" * 60, fg="blue"))

    # Validate input
    if not input_path.exists():
        click.echo(click.style(f"Error: Input file not found: {input_path}", fg="red"))
        sys.exit(1)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load data
    click.echo(f"\n[1/5] Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    click.echo(f"      Loaded {len(df):,} rows")

    # Step 2: Clean data
    click.echo("\n[2/5] Cleaning transactions...")
    df_clean = clean_transactions(df)
    click.echo(f"      {len(df_clean):,} rows after cleaning")

    # Step 3: Calculate RFM
    click.echo("\n[3/5] Calculating RFM features...")
    from src.config import DATE_COL

    df_clean[DATE_COL] = pd.to_datetime(df_clean[DATE_COL])
    reference_date = df_clean[DATE_COL].max() + pd.Timedelta(days=1)

    calculator = RFMCalculator(reference_date=reference_date)
    rfm_df = calculator.fit_transform(df_clean)
    click.echo(f"      {len(rfm_df):,} customers with RFM features")

    if verbose:
        click.echo("\n      RFM Statistics:")
        stats = calculator.get_statistics()
        click.echo(stats.round(2).to_string().replace("\n", "\n      "))

    # Step 4: Train model
    click.echo(f"\n[4/5] Training KMeans with {n_clusters} clusters...")
    segmenter = CustomerSegmenter(n_clusters=n_clusters, random_state=random_state)
    labels = segmenter.fit_predict(rfm_df)
    rfm_df["segment"] = labels

    # Evaluate
    X_scaled = segmenter.scaler.transform(rfm_df[["recency", "frequency", "monetary"]])
    metrics = evaluate_clustering(X_scaled, labels)

    click.echo(f"      Silhouette Score: {metrics['silhouette']:.3f}")
    click.echo(f"      Calinski-Harabasz: {metrics['calinski_harabasz']:.1f}")
    click.echo(f"      Davies-Bouldin: {metrics['davies_bouldin']:.3f}")

    # Distribution
    click.echo("\n      Segment distribution:")
    for seg_id in sorted(rfm_df["segment"].unique()):
        count = (rfm_df["segment"] == seg_id).sum()
        pct = count / len(rfm_df) * 100
        click.echo(f"        Segment {seg_id}: {count:,} ({pct:.1f}%)")

    # Step 5: Save
    click.echo("\n[5/5] Saving outputs...")

    # Save model
    segmenter.save(output_path)
    click.echo(f"      Model saved to {output_path}")

    # Save RFM data
    rfm_parquet = PROCESSED_DATA_DIR / "customers_rfm.parquet"
    rfm_csv = PROCESSED_DATA_DIR / "customers_rfm.csv"
    rfm_df.to_parquet(rfm_parquet)
    rfm_df.to_csv(rfm_csv)
    click.echo(f"      RFM data saved to {rfm_parquet}")

    click.echo(click.style("\n" + "=" * 60, fg="green"))
    click.echo(click.style("  Training completed successfully!", fg="green", bold=True))
    click.echo(click.style("=" * 60, fg="green"))


@main.command()
@click.option(
    "--model",
    "-m",
    "model_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to model directory.",
)
@click.option(
    "--input",
    "-i",
    "input_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to RFM data file (CSV or Parquet).",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to save predictions.",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output.")
def predict(
    model_path: Path | None,
    input_path: Path,
    output_path: Path | None,
    verbose: bool,
) -> None:
    """Predict customer segments using a trained model.

    The input file should contain RFM features (recency, frequency, monetary).
    """
    from src.models.clustering import CustomerSegmenter

    # Default model path
    if model_path is None:
        model_path = MODELS_DIR

    click.echo(click.style("Predicting customer segments...", fg="blue", bold=True))

    # Load model
    click.echo(f"\n[1/3] Loading model from {model_path}...")
    segmenter = CustomerSegmenter.load(model_path)

    # Load data
    click.echo(f"[2/3] Loading data from {input_path}...")
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path, index_col=0)

    click.echo(f"      Loaded {len(df):,} customers")

    # Predict
    click.echo("[3/3] Predicting segments...")
    features = df[["recency", "frequency", "monetary"]]
    labels = segmenter.predict(features)
    df["segment"] = labels

    # Output
    if output_path:
        if output_path.suffix == ".parquet":
            df.to_parquet(output_path)
        else:
            df.to_csv(output_path)
        click.echo(f"\n      Predictions saved to {output_path}")
    else:
        click.echo("\nSegment distribution:")
        for seg_id in sorted(df["segment"].unique()):
            count = (df["segment"] == seg_id).sum()
            pct = count / len(df) * 100
            click.echo(f"  Segment {seg_id}: {count:,} ({pct:.1f}%)")

    click.echo(click.style("\nPrediction completed!", fg="green", bold=True))


@main.command()
@click.option(
    "--model",
    "-m",
    "model_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to model directory.",
)
@click.option(
    "--data",
    "-d",
    "data_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to RFM data file for evaluation.",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output.")
def evaluate(
    model_path: Path | None,
    data_path: Path | None,
    verbose: bool,
) -> None:
    """Evaluate the clustering model.

    Computes various clustering metrics on the provided data.
    """
    from src.models.clustering import CustomerSegmenter
    from src.models.evaluation import evaluate_clustering

    # Default paths
    if model_path is None:
        model_path = MODELS_DIR
    if data_path is None:
        data_path = PROCESSED_DATA_DIR / "customers_rfm.parquet"

    click.echo(click.style("Evaluating model...", fg="blue", bold=True))

    # Validate paths
    if not data_path.exists():
        click.echo(click.style(f"Error: Data file not found: {data_path}", fg="red"))
        sys.exit(1)

    # Load model and data
    click.echo("\n[1/2] Loading model and data...")
    segmenter = CustomerSegmenter.load(model_path)

    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, index_col=0)

    click.echo(f"      Loaded {len(df):,} customers")

    # Evaluate
    click.echo("[2/2] Computing metrics...")
    features = df[["recency", "frequency", "monetary"]]
    X_scaled = segmenter.scaler.transform(features)

    if "segment" in df.columns:
        labels = df["segment"].values
    else:
        labels = segmenter.predict(features)

    metrics = evaluate_clustering(X_scaled, labels)

    # Display results
    click.echo(click.style("\n" + "=" * 40, fg="cyan"))
    click.echo(click.style("  Clustering Metrics", fg="cyan", bold=True))
    click.echo(click.style("=" * 40, fg="cyan"))
    click.echo(f"  Silhouette Score:    {metrics['silhouette']:.4f}")
    click.echo(f"  Calinski-Harabasz:   {metrics['calinski_harabasz']:.2f}")
    click.echo(f"  Davies-Bouldin:      {metrics['davies_bouldin']:.4f}")
    click.echo(f"  Number of clusters:  {metrics['n_clusters']}")
    click.echo(click.style("=" * 40, fg="cyan"))

    if verbose:
        click.echo("\nSegment summary:")
        summary = segmenter.get_segment_summary(features)
        click.echo(summary.to_string())


@main.command()
@click.option(
    "--port",
    "-p",
    default=8501,
    type=int,
    help="Port to run the dashboard on (default: 8501).",
)
@click.option(
    "--host",
    "-h",
    default="localhost",
    help="Host to bind the dashboard to (default: localhost).",
)
def serve(port: int, host: str) -> None:
    """Launch the Streamlit dashboard.

    This command starts the interactive web dashboard for exploring
    customer segments.
    """
    import subprocess

    click.echo(click.style("Starting Streamlit dashboard...", fg="blue", bold=True))
    click.echo(f"\n  URL: http://{host}:{port}")
    click.echo("  Press Ctrl+C to stop\n")

    # Get the app path
    from src.config import ROOT_DIR

    app_path = ROOT_DIR / "app" / "app.py"

    # Launch streamlit
    try:
        subprocess.run(
            [
                "streamlit",
                "run",
                str(app_path),
                f"--server.port={port}",
                f"--server.address={host}",
            ],
            check=True,
        )
    except KeyboardInterrupt:
        click.echo("\nDashboard stopped.")
    except subprocess.CalledProcessError as e:
        click.echo(click.style(f"Error starting dashboard: {e}", fg="red"))
        sys.exit(1)


@main.command()
@click.option("--verbose", "-v", is_flag=True, help="Show detailed information.")
def info(verbose: bool) -> None:
    """Display project information and paths."""
    from src.config import (
        KMEANS_MODEL_FILE,
        PROCESSED_RFM_FILE,
        RAW_TRANSACTIONS_FILE,
        ROOT_DIR,
        SCALER_FILE,
    )

    click.echo(click.style("\nOlist Customer Segmentation", fg="cyan", bold=True))
    click.echo(click.style("=" * 40, fg="cyan"))

    click.echo(f"\nProject root: {ROOT_DIR}")

    # Check data files
    click.echo("\nData files:")
    files = [
        ("Raw transactions", RAW_TRANSACTIONS_FILE),
        ("Processed RFM", PROCESSED_RFM_FILE),
        ("KMeans model", KMEANS_MODEL_FILE),
        ("Scaler", SCALER_FILE),
    ]

    for name, path in files:
        exists = path.exists()
        status = (
            click.style("OK", fg="green")
            if exists
            else click.style("MISSING", fg="red")
        )
        click.echo(f"  [{status}] {name}")
        if verbose:
            click.echo(f"       {path}")

    if verbose:
        click.echo("\nConfiguration:")
        click.echo(f"  N_CLUSTERS: {N_CLUSTERS}")
        click.echo(f"  RANDOM_STATE: {RANDOM_STATE}")


if __name__ == "__main__":
    main()
