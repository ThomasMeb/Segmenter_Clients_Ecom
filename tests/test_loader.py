"""
Tests pour le module src.data.loader.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data.loader import load_transactions, load_rfm_data, validate_dataframe


class TestLoadTransactions:
    """Tests pour la fonction load_transactions."""

    def test_load_transactions_from_csv(self, test_data_dir, sample_transactions):
        """Test du chargement d'un fichier CSV valide."""
        # Créer un fichier CSV de test
        csv_path = test_data_dir / "test_transactions.csv"
        sample_transactions.to_csv(csv_path, index=False)

        # Charger et vérifier
        df = load_transactions(csv_path)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_transactions)
        assert "customer_unique_id" in df.columns
        assert "order_id" in df.columns
        assert "price" in df.columns

    def test_load_transactions_parses_dates(self, test_data_dir, sample_transactions):
        """Test que les dates sont parsées correctement."""
        csv_path = test_data_dir / "test_transactions.csv"
        sample_transactions.to_csv(csv_path, index=False)

        df = load_transactions(csv_path, parse_dates=True)

        assert pd.api.types.is_datetime64_any_dtype(df["order_purchase_timestamp"])

    def test_load_transactions_removes_unnamed_column(self, test_data_dir, sample_transactions):
        """Test que la colonne Unnamed: 0 est supprimée."""
        csv_path = test_data_dir / "test_transactions.csv"
        # Sauvegarder avec index pour créer Unnamed: 0
        sample_transactions.to_csv(csv_path, index=True)

        df = load_transactions(csv_path)

        assert "Unnamed: 0" not in df.columns

    def test_load_transactions_file_not_found(self):
        """Test que FileNotFoundError est levée pour un fichier inexistant."""
        with pytest.raises(FileNotFoundError):
            load_transactions("/nonexistent/path/file.csv")

    def test_load_transactions_missing_columns(self, test_data_dir):
        """Test que ValueError est levée si des colonnes sont manquantes."""
        # Créer un CSV avec colonnes manquantes
        csv_path = test_data_dir / "incomplete.csv"
        pd.DataFrame({"col1": [1, 2], "col2": [3, 4]}).to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="Colonnes manquantes"):
            load_transactions(csv_path)


class TestLoadRfmData:
    """Tests pour la fonction load_rfm_data."""

    def test_load_rfm_from_csv(self, test_data_dir, sample_rfm):
        """Test du chargement RFM depuis CSV."""
        csv_path = test_data_dir / "rfm.csv"
        sample_rfm.to_csv(csv_path)

        df = load_rfm_data(csv_path)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_rfm)

    def test_load_rfm_from_parquet(self, test_data_dir, sample_rfm):
        """Test du chargement RFM depuis Parquet."""
        parquet_path = test_data_dir / "rfm.parquet"
        sample_rfm.to_parquet(parquet_path)

        df = load_rfm_data(parquet_path)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_rfm)

    def test_load_rfm_file_not_found(self):
        """Test que FileNotFoundError est levée pour un fichier inexistant."""
        with pytest.raises(FileNotFoundError):
            load_rfm_data("/nonexistent/path/rfm.parquet")

    def test_load_rfm_unsupported_format(self, test_data_dir):
        """Test que ValueError est levée pour un format non supporté."""
        txt_path = test_data_dir / "rfm.txt"
        txt_path.write_text("dummy content")

        with pytest.raises(ValueError, match="Format non supporté"):
            load_rfm_data(txt_path)


class TestValidateDataframe:
    """Tests pour la fonction validate_dataframe."""

    def test_validate_success(self, sample_transactions):
        """Test validation réussie."""
        result = validate_dataframe(
            sample_transactions,
            required_columns=["customer_unique_id", "order_id"],
            name="Test"
        )
        assert result is True

    def test_validate_missing_columns(self, sample_transactions):
        """Test validation avec colonnes manquantes."""
        with pytest.raises(ValueError, match="colonnes manquantes"):
            validate_dataframe(
                sample_transactions,
                required_columns=["nonexistent_column"],
                name="Test"
            )
