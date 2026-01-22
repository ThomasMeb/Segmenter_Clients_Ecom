"""
Tests pour le module src.features.rfm.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.features.rfm import RFMCalculator, calculate_rfm


class TestRFMCalculator:
    """Tests pour la classe RFMCalculator."""

    def test_init_with_datetime(self):
        """Test d'initialisation avec une date datetime."""
        ref_date = datetime(2018, 9, 1)
        calculator = RFMCalculator(reference_date=ref_date)

        assert calculator.reference_date == ref_date
        assert calculator.rfm_data is None

    def test_init_with_string(self):
        """Test d'initialisation avec une date string."""
        calculator = RFMCalculator(reference_date="2018-09-01")

        assert calculator.reference_date == pd.Timestamp("2018-09-01")

    def test_init_without_date(self):
        """Test d'initialisation sans date de référence."""
        calculator = RFMCalculator()

        assert calculator.reference_date is None

    def test_fit_transform_basic(self, sample_transactions):
        """Test de base de fit_transform."""
        calculator = RFMCalculator(reference_date="2018-04-15")
        rfm = calculator.fit_transform(sample_transactions)

        # Vérifier la structure
        assert isinstance(rfm, pd.DataFrame)
        assert list(rfm.columns) == ["recency", "frequency", "monetary"]
        assert rfm.index.name == "customer_unique_id"

        # Vérifier les types
        assert rfm["recency"].dtype in [np.int64, np.int32, int]
        assert rfm["frequency"].dtype in [np.int64, np.int32, int]
        assert np.issubdtype(rfm["monetary"].dtype, np.floating)

    def test_fit_transform_values(self, sample_transactions):
        """Test des valeurs calculées."""
        calculator = RFMCalculator(reference_date="2018-04-15")
        rfm = calculator.fit_transform(sample_transactions)

        # Tous les recency doivent être >= 0
        assert (rfm["recency"] >= 0).all()

        # Tous les frequency doivent être >= 1
        assert (rfm["frequency"] >= 1).all()

        # Tous les monetary doivent être > 0
        assert (rfm["monetary"] > 0).all()

    def test_fit_transform_auto_reference_date(self, sample_transactions):
        """Test que la date de référence est auto-calculée si non fournie."""
        calculator = RFMCalculator()
        rfm = calculator.fit_transform(sample_transactions)

        # La date de référence doit être définie après fit_transform
        assert calculator.reference_date is not None
        assert calculator.reference_date == sample_transactions["order_purchase_timestamp"].max()

    def test_get_statistics(self, sample_transactions):
        """Test de la méthode get_statistics."""
        calculator = RFMCalculator(reference_date="2018-04-15")
        calculator.fit_transform(sample_transactions)

        stats = calculator.get_statistics()

        assert isinstance(stats, pd.DataFrame)
        assert "mean" in stats.index
        assert "std" in stats.index
        assert "min" in stats.index
        assert "max" in stats.index

    def test_get_statistics_without_fit(self):
        """Test que get_statistics lève une erreur sans fit_transform."""
        calculator = RFMCalculator()

        with pytest.raises(ValueError, match="fit_transform"):
            calculator.get_statistics()

    def test_save_parquet(self, sample_transactions, test_data_dir):
        """Test de sauvegarde en Parquet."""
        calculator = RFMCalculator(reference_date="2018-04-15")
        calculator.fit_transform(sample_transactions)

        output_path = test_data_dir / "rfm_test.parquet"
        calculator.save(str(output_path), format="parquet")

        assert output_path.exists()

        # Vérifier le contenu
        loaded = pd.read_parquet(output_path)
        assert len(loaded) == len(calculator.rfm_data)

    def test_save_csv(self, sample_transactions, test_data_dir):
        """Test de sauvegarde en CSV."""
        calculator = RFMCalculator(reference_date="2018-04-15")
        calculator.fit_transform(sample_transactions)

        output_path = test_data_dir / "rfm_test.csv"
        calculator.save(str(output_path), format="csv")

        assert output_path.exists()

    def test_save_without_fit(self, test_data_dir):
        """Test que save lève une erreur sans fit_transform."""
        calculator = RFMCalculator()

        with pytest.raises(ValueError, match="fit_transform"):
            calculator.save(str(test_data_dir / "rfm.parquet"))

    def test_save_invalid_format(self, sample_transactions, test_data_dir):
        """Test que save lève une erreur pour un format invalide."""
        calculator = RFMCalculator(reference_date="2018-04-15")
        calculator.fit_transform(sample_transactions)

        with pytest.raises(ValueError, match="Format non supporté"):
            calculator.save(str(test_data_dir / "rfm.txt"), format="txt")


class TestCalculateRfm:
    """Tests pour la fonction calculate_rfm."""

    def test_calculate_rfm_basic(self, sample_transactions):
        """Test de base de la fonction calculate_rfm."""
        rfm = calculate_rfm(sample_transactions, reference_date=datetime(2018, 4, 15))

        assert isinstance(rfm, pd.DataFrame)
        assert list(rfm.columns) == ["recency", "frequency", "monetary"]

    def test_calculate_rfm_custom_columns(self, sample_transactions):
        """Test avec des noms de colonnes personnalisés."""
        # Renommer les colonnes
        df = sample_transactions.rename(columns={
            "customer_unique_id": "client_id",
            "order_purchase_timestamp": "date",
            "price": "amount"
        })

        rfm = calculate_rfm(
            df,
            customer_col="client_id",
            date_col="date",
            amount_col="amount",
            reference_date=datetime(2018, 4, 15)
        )

        assert isinstance(rfm, pd.DataFrame)
        assert len(rfm) > 0

    def test_calculate_rfm_aggregation(self, sample_transactions):
        """Test que l'agrégation fonctionne correctement."""
        rfm = calculate_rfm(sample_transactions, reference_date=datetime(2018, 4, 15))

        # Nombre de clients uniques dans l'entrée
        n_unique_customers = sample_transactions["customer_unique_id"].nunique()

        # Doit correspondre au nombre de lignes dans RFM
        assert len(rfm) == n_unique_customers
