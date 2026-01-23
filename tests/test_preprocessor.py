"""Tests pour le module de prétraitement des données."""

import numpy as np
import pandas as pd
import pytest

from src.config import AMOUNT_COL, CUSTOMER_ID_COL, DATE_COL, ORDER_ID_COL
from src.data.preprocessor import (
    clean_transactions,
    prepare_for_rfm,
    remove_outliers,
)


@pytest.fixture
def sample_transactions():
    """Crée un DataFrame de transactions de test."""
    return pd.DataFrame(
        {
            CUSTOMER_ID_COL: ["c1", "c2", "c3", "c1", "c2", "c3", "c4", "c4"],
            ORDER_ID_COL: ["o1", "o2", "o3", "o4", "o5", "o6", "o7", "o8"],
            DATE_COL: [
                "2023-01-01",
                "2023-01-02",
                "2023-01-03",
                "2023-01-04",
                "2023-01-05",
                "2023-01-06",
                "2023-01-07",
                "2023-01-08",
            ],
            AMOUNT_COL: [100.0, 200.0, 150.0, 50.0, 75.0, 300.0, 25.0, 125.0],
        }
    )


@pytest.fixture
def transactions_with_issues():
    """Crée un DataFrame avec des problèmes de qualité."""
    return pd.DataFrame(
        {
            CUSTOMER_ID_COL: ["c1", "c2", None, "c1", "c2", "c3"],
            ORDER_ID_COL: ["o1", "o2", "o3", "o1", "o5", None],
            DATE_COL: [
                "2023-01-01",
                "2023-01-02",
                "2023-01-03",
                "2023-01-01",
                "2023-01-05",
                "2023-01-06",
            ],
            AMOUNT_COL: [100.0, -50.0, 150.0, 100.0, 0.0, 200.0],
        }
    )


class TestCleanTransactions:
    """Tests pour la fonction clean_transactions."""

    def test_clean_transactions_basic(self, sample_transactions):
        """Test du nettoyage basique."""
        result = clean_transactions(sample_transactions)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_transactions)

    def test_removes_duplicates(self, transactions_with_issues):
        """Test de suppression des doublons."""
        result = clean_transactions(transactions_with_issues)
        # La ligne dupliquée (c1, o1) doit être supprimée
        assert len(result) < len(transactions_with_issues)

    def test_removes_null_customer_id(self, transactions_with_issues):
        """Test de suppression des customer_id null."""
        result = clean_transactions(transactions_with_issues)
        assert result[CUSTOMER_ID_COL].isnull().sum() == 0

    def test_removes_null_order_id(self, transactions_with_issues):
        """Test de suppression des order_id null."""
        result = clean_transactions(transactions_with_issues)
        assert result[ORDER_ID_COL].isnull().sum() == 0

    def test_removes_negative_amounts(self, transactions_with_issues):
        """Test de suppression des montants négatifs."""
        result = clean_transactions(transactions_with_issues)
        assert (result[AMOUNT_COL] > 0).all()

    def test_removes_zero_amounts(self, transactions_with_issues):
        """Test de suppression des montants nuls."""
        result = clean_transactions(transactions_with_issues)
        assert (result[AMOUNT_COL] > 0).all()

    def test_converts_date_column(self, sample_transactions):
        """Test de conversion de la colonne date."""
        result = clean_transactions(sample_transactions)
        assert pd.api.types.is_datetime64_any_dtype(result[DATE_COL])

    def test_resets_index(self, transactions_with_issues):
        """Test que l'index est réinitialisé."""
        result = clean_transactions(transactions_with_issues)
        assert list(result.index) == list(range(len(result)))

    def test_preserves_valid_data(self, sample_transactions):
        """Test que les données valides sont préservées."""
        result = clean_transactions(sample_transactions)
        assert len(result) == len(sample_transactions)
        assert result[AMOUNT_COL].sum() == sample_transactions[AMOUNT_COL].sum()


class TestPrepareForRfm:
    """Tests pour la fonction prepare_for_rfm."""

    def test_prepare_for_rfm_basic(self, sample_transactions):
        """Test de la préparation basique."""
        result = prepare_for_rfm(sample_transactions)
        assert isinstance(result, pd.DataFrame)

    def test_selects_required_columns(self, sample_transactions):
        """Test de sélection des colonnes requises."""
        result = prepare_for_rfm(sample_transactions)
        expected_cols = [CUSTOMER_ID_COL, DATE_COL, AMOUNT_COL]
        assert list(result.columns) == expected_cols

    def test_converts_date(self, sample_transactions):
        """Test de conversion de la date."""
        result = prepare_for_rfm(sample_transactions)
        assert pd.api.types.is_datetime64_any_dtype(result[DATE_COL])

    def test_sorts_by_date(self, sample_transactions):
        """Test du tri par date."""
        result = prepare_for_rfm(sample_transactions)
        dates = result[DATE_COL].tolist()
        assert dates == sorted(dates)

    def test_handles_already_datetime(self, sample_transactions):
        """Test avec une colonne date déjà en datetime."""
        sample_transactions[DATE_COL] = pd.to_datetime(sample_transactions[DATE_COL])
        result = prepare_for_rfm(sample_transactions)
        assert pd.api.types.is_datetime64_any_dtype(result[DATE_COL])


class TestRemoveOutliers:
    """Tests pour la fonction remove_outliers."""

    @pytest.fixture
    def df_with_outliers(self):
        """Crée un DataFrame avec des outliers."""
        np.random.seed(42)
        normal_values = np.random.normal(100, 10, 100)
        outliers = [500, 600, -100]  # Outliers évidents
        values = np.concatenate([normal_values, outliers])
        return pd.DataFrame({"value": values})

    def test_remove_outliers_iqr(self, df_with_outliers):
        """Test de suppression des outliers par IQR."""
        result = remove_outliers(df_with_outliers, "value", method="iqr")
        assert len(result) < len(df_with_outliers)

    def test_remove_outliers_zscore(self, df_with_outliers):
        """Test de suppression des outliers par zscore."""
        result = remove_outliers(
            df_with_outliers, "value", method="zscore", threshold=3
        )
        assert len(result) < len(df_with_outliers)

    def test_iqr_removes_extreme_values(self, df_with_outliers):
        """Test que IQR supprime les valeurs extrêmes."""
        result = remove_outliers(df_with_outliers, "value", method="iqr")
        assert result["value"].max() < 500
        assert result["value"].min() > -100

    def test_zscore_removes_extreme_values(self, df_with_outliers):
        """Test que zscore supprime les valeurs extrêmes."""
        result = remove_outliers(
            df_with_outliers, "value", method="zscore", threshold=3
        )
        assert result["value"].max() < 500

    def test_invalid_method_raises_error(self, df_with_outliers):
        """Test qu'une méthode invalide lève une erreur."""
        with pytest.raises(ValueError, match="Méthode inconnue"):
            remove_outliers(df_with_outliers, "value", method="invalid")

    def test_custom_threshold_iqr(self, df_with_outliers):
        """Test avec un seuil IQR personnalisé."""
        result_default = remove_outliers(
            df_with_outliers, "value", method="iqr", threshold=1.5
        )
        result_strict = remove_outliers(
            df_with_outliers, "value", method="iqr", threshold=1.0
        )
        # Un seuil plus strict devrait supprimer plus de valeurs
        assert len(result_strict) <= len(result_default)

    def test_custom_threshold_zscore(self, df_with_outliers):
        """Test avec un seuil zscore personnalisé."""
        result_default = remove_outliers(
            df_with_outliers, "value", method="zscore", threshold=3
        )
        result_strict = remove_outliers(
            df_with_outliers, "value", method="zscore", threshold=2
        )
        # Un seuil plus strict devrait supprimer plus de valeurs
        assert len(result_strict) <= len(result_default)

    def test_preserves_non_outlier_data(self):
        """Test que les données non-outliers sont préservées."""
        df = pd.DataFrame({"value": [10, 20, 30, 40, 50]})
        result = remove_outliers(df, "value", method="iqr")
        assert len(result) == len(df)

    def test_returns_copy(self, df_with_outliers):
        """Test que la fonction retourne une copie."""
        result = remove_outliers(df_with_outliers, "value", method="iqr")
        result["value"] = 0
        assert df_with_outliers["value"].sum() != 0
