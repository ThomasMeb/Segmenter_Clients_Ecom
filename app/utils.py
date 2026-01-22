"""
Utilitaires pour le dashboard Streamlit.

Ce module gère le chargement des données et la génération
de données de démonstration si nécessaire.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def get_project_root() -> Path:
    """Retourne le chemin racine du projet."""
    return Path(__file__).parent.parent


def generate_demo_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Génère des données RFM de démonstration.

    Utilisé quand les vraies données ne sont pas disponibles
    (ex: déploiement Streamlit Cloud sans Git LFS).

    Parameters
    ----------
    n_samples : int
        Nombre de clients à générer.
    random_state : int
        Graine aléatoire pour la reproductibilité.

    Returns
    -------
    pd.DataFrame
        DataFrame avec colonnes recency, frequency, monetary, segment.
    """
    np.random.seed(random_state)

    # Générer des données réalistes par segment
    segments = []

    # Segment 0 : Clients Récents (54%)
    n_recent = int(n_samples * 0.54)
    recent = pd.DataFrame({
        "recency": np.random.randint(1, 90, n_recent),
        "frequency": np.ones(n_recent, dtype=int),
        "monetary": np.random.exponential(100, n_recent) + 20,
        "segment": 0,
    })
    segments.append(recent)

    # Segment 1 : Clients Fidèles (3%)
    n_loyal = int(n_samples * 0.03)
    loyal = pd.DataFrame({
        "recency": np.random.randint(10, 150, n_loyal),
        "frequency": np.random.randint(3, 10, n_loyal),
        "monetary": np.random.exponential(200, n_loyal) + 100,
        "segment": 1,
    })
    segments.append(loyal)

    # Segment 2 : Clients Dormants (40%)
    n_dormant = int(n_samples * 0.40)
    dormant = pd.DataFrame({
        "recency": np.random.randint(180, 400, n_dormant),
        "frequency": np.ones(n_dormant, dtype=int),
        "monetary": np.random.exponential(80, n_dormant) + 15,
        "segment": 2,
    })
    segments.append(dormant)

    # Segment 3 : Clients VIP (3%)
    n_vip = n_samples - n_recent - n_loyal - n_dormant
    vip = pd.DataFrame({
        "recency": np.random.randint(1, 60, n_vip),
        "frequency": np.random.randint(5, 15, n_vip),
        "monetary": np.random.exponential(500, n_vip) + 200,
        "segment": 3,
    })
    segments.append(vip)

    # Combiner et mélanger
    df = pd.concat(segments, ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Ajouter un index client
    df.index = [f"customer_{i:05d}" for i in range(len(df))]
    df.index.name = "customer_unique_id"

    return df


def load_rfm_data() -> tuple[pd.DataFrame, bool]:
    """
    Charge les données RFM depuis le fichier ou génère des données de démo.

    Returns
    -------
    tuple[pd.DataFrame, bool]
        (DataFrame RFM, True si données réelles / False si démo)
    """
    root = get_project_root()

    # Essayer de charger les données réelles
    data_paths = [
        root / "data" / "processed" / "customers_rfm.parquet",
        root / "data" / "processed" / "customers_rfm.csv",
        root / "data" / "processed" / "data_RFM.csv",
    ]

    for path in data_paths:
        if path.exists():
            try:
                if path.suffix == ".parquet":
                    df = pd.read_parquet(path)
                else:
                    df = pd.read_csv(path, index_col=0)

                # Vérifier que les colonnes nécessaires sont présentes
                required_cols = ["recency", "frequency", "monetary"]
                if all(col in df.columns for col in required_cols):
                    return df, True
            except Exception:
                continue

    # Générer des données de démonstration
    return generate_demo_data(n_samples=5000), False
