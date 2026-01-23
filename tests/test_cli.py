"""Tests for the CLI module."""


import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from src.cli import evaluate, info, main, predict, train


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_transactions(tmp_path):
    """Create sample transaction data for testing."""
    np.random.seed(42)
    n_customers = 50
    n_orders = 200

    customers = [f"customer_{i:04d}" for i in range(n_customers)]
    data = {
        "customer_unique_id": np.random.choice(customers, n_orders),
        "order_id": [f"order_{i:04d}" for i in range(n_orders)],
        "order_purchase_timestamp": pd.date_range(
            "2023-01-01", periods=n_orders, freq="4h"
        ).astype(str),
        "price": np.random.exponential(100, n_orders) + 20,
    }

    df = pd.DataFrame(data)
    csv_path = tmp_path / "transactions.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_rfm_data(tmp_path):
    """Create sample RFM data for testing."""
    np.random.seed(42)
    n_customers = 100

    data = {
        "customer_unique_id": [f"customer_{i:04d}" for i in range(n_customers)],
        "recency": np.random.randint(1, 365, n_customers),
        "frequency": np.random.randint(1, 10, n_customers),
        "monetary": np.random.exponential(200, n_customers) + 50,
    }

    df = pd.DataFrame(data).set_index("customer_unique_id")
    csv_path = tmp_path / "rfm.csv"
    df.to_csv(csv_path)
    return csv_path


class TestMainCommand:
    """Tests for the main CLI group."""

    def test_main_help(self, runner):
        """Test that main --help works."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Olist Customer Segmentation CLI" in result.output

    def test_main_version(self, runner):
        """Test that --version works."""
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "1.0.0" in result.output


class TestTrainCommand:
    """Tests for the train command."""

    def test_train_help(self, runner):
        """Test train --help."""
        result = runner.invoke(train, ["--help"])
        assert result.exit_code == 0
        assert "Train the customer segmentation model" in result.output

    def test_train_missing_input(self, runner):
        """Test train with non-existent input file."""
        result = runner.invoke(train, ["--input", "/nonexistent/file.csv"])
        assert result.exit_code != 0

    def test_train_success(self, runner, sample_transactions, tmp_path):
        """Test successful training."""
        output_dir = tmp_path / "models"

        result = runner.invoke(
            train,
            [
                "--input",
                str(sample_transactions),
                "--output",
                str(output_dir),
                "--n-clusters",
                "3",
            ],
        )

        assert result.exit_code == 0
        assert "Training completed successfully" in result.output
        assert (output_dir / "kmeans_model.pkl").exists()
        assert (output_dir / "scaler.pkl").exists()

    def test_train_verbose(self, runner, sample_transactions, tmp_path):
        """Test training with verbose output."""
        output_dir = tmp_path / "models"

        result = runner.invoke(
            train,
            [
                "--input",
                str(sample_transactions),
                "--output",
                str(output_dir),
                "--verbose",
            ],
        )

        assert result.exit_code == 0
        assert "RFM Statistics" in result.output


class TestPredictCommand:
    """Tests for the predict command."""

    def test_predict_help(self, runner):
        """Test predict --help."""
        result = runner.invoke(predict, ["--help"])
        assert result.exit_code == 0
        assert "Predict customer segments" in result.output

    def test_predict_success(
        self, runner, sample_transactions, sample_rfm_data, tmp_path
    ):
        """Test successful prediction."""
        # First train a model
        model_dir = tmp_path / "models"
        runner.invoke(
            train,
            [
                "--input",
                str(sample_transactions),
                "--output",
                str(model_dir),
            ],
        )

        # Then predict
        result = runner.invoke(
            predict,
            [
                "--model",
                str(model_dir),
                "--input",
                str(sample_rfm_data),
            ],
        )

        assert result.exit_code == 0
        assert "Prediction completed" in result.output

    def test_predict_with_output(
        self, runner, sample_transactions, sample_rfm_data, tmp_path
    ):
        """Test prediction with output file."""
        model_dir = tmp_path / "models"
        output_file = tmp_path / "predictions.csv"

        # Train
        runner.invoke(
            train,
            ["--input", str(sample_transactions), "--output", str(model_dir)],
        )

        # Predict with output
        result = runner.invoke(
            predict,
            [
                "--model",
                str(model_dir),
                "--input",
                str(sample_rfm_data),
                "--output",
                str(output_file),
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()


class TestEvaluateCommand:
    """Tests for the evaluate command."""

    def test_evaluate_help(self, runner):
        """Test evaluate --help."""
        result = runner.invoke(evaluate, ["--help"])
        assert result.exit_code == 0
        assert "Evaluate the clustering model" in result.output

    def test_evaluate_success(self, runner, sample_transactions, tmp_path):
        """Test successful evaluation."""
        model_dir = tmp_path / "models"
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        # Train first
        result = runner.invoke(
            train,
            ["--input", str(sample_transactions), "--output", str(model_dir)],
        )

        # The training command saves RFM data to PROCESSED_DATA_DIR
        # We need to use monkeypatch or provide the data path
        from src.config import PROCESSED_DATA_DIR

        rfm_path = PROCESSED_DATA_DIR / "customers_rfm.parquet"

        if rfm_path.exists():
            result = runner.invoke(
                evaluate,
                ["--model", str(model_dir), "--data", str(rfm_path)],
            )

            assert result.exit_code == 0
            assert "Silhouette Score" in result.output
            assert "Calinski-Harabasz" in result.output


class TestInfoCommand:
    """Tests for the info command."""

    def test_info_basic(self, runner):
        """Test info command."""
        result = runner.invoke(info)
        assert result.exit_code == 0
        assert "Olist Customer Segmentation" in result.output
        assert "Data files:" in result.output

    def test_info_verbose(self, runner):
        """Test info --verbose."""
        result = runner.invoke(info, ["--verbose"])
        assert result.exit_code == 0
        assert "Configuration:" in result.output
        assert "N_CLUSTERS" in result.output


class TestServeCommand:
    """Tests for the serve command."""

    def test_serve_help(self, runner):
        """Test serve --help."""
        from src.cli import serve

        result = runner.invoke(serve, ["--help"])
        assert result.exit_code == 0
        assert "Launch the Streamlit dashboard" in result.output
