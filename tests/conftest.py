"""
Fixtures pytest pour les tests.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_transactions():
    """Génère un DataFrame de transactions de test."""
    np.random.seed(42)
    n_samples = 100

    return pd.DataFrame({
        "customer_unique_id": [f"customer_{i % 20}" for i in range(n_samples)],
        "order_id": [f"order_{i}" for i in range(n_samples)],
        "price": np.random.uniform(10, 500, n_samples),
        "order_purchase_timestamp": pd.date_range(
            start="2018-01-01",
            periods=n_samples,
            freq="D"
        ),
    })


@pytest.fixture
def sample_rfm():
    """Génère un DataFrame RFM de test."""
    np.random.seed(42)
    n_samples = 50

    return pd.DataFrame({
        "recency": np.random.randint(1, 365, n_samples),
        "frequency": np.random.randint(1, 10, n_samples),
        "monetary": np.random.uniform(50, 1000, n_samples),
    }, index=[f"customer_{i}" for i in range(n_samples)])


@pytest.fixture
def test_data_dir(tmp_path):
    """Crée un répertoire temporaire pour les tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir
