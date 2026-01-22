#!/usr/bin/env python3
"""
Script pour préparer les données du dashboard.

Ce script génère les données RFM, entraîne le modèle de segmentation
et sauvegarde tous les fichiers nécessaires au dashboard.

Usage:
    python scripts/prepare_dashboard_data.py
"""

import sys
from pathlib import Path

# Ajouter le répertoire racine au path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import numpy as np

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    N_CLUSTERS,
    RANDOM_STATE,
)
from src.features.rfm import RFMCalculator
from src.models.clustering import CustomerSegmenter


def load_raw_data() -> pd.DataFrame:
    """Charge les données brutes."""
    # Essayer différentes sources de données
    sources = [
        RAW_DATA_DIR / "data.csv",
        RAW_DATA_DIR / "transactions_clean.csv",
        ROOT_DIR / "data" / "olist_orders_dataset.csv",
    ]

    for source in sources:
        if source.exists():
            print(f"Chargement depuis: {source}")
            df = pd.read_csv(source)

            # Nettoyer
            if "Unnamed: 0" in df.columns:
                df = df.drop(columns=["Unnamed: 0"])

            return df

    raise FileNotFoundError(f"Aucune source de données trouvée dans {sources}")


def prepare_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Prépare les transactions pour le calcul RFM."""
    # Vérifier les colonnes nécessaires
    required_cols = ["customer_unique_id", "order_id", "price", "order_purchase_timestamp"]

    # Renommer si nécessaire
    rename_map = {
        "order_purchase_timestamp": "order_purchase_timestamp",
    }

    if all(col in df.columns for col in required_cols):
        return df[required_cols].copy()

    print(f"Colonnes disponibles: {df.columns.tolist()}")
    raise ValueError(f"Colonnes manquantes. Requis: {required_cols}")


def main():
    """Fonction principale."""
    print("=" * 60)
    print("Préparation des données pour le dashboard")
    print("=" * 60)

    # Créer les dossiers si nécessaire
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Charger les données
    print("\n[1/4] Chargement des données...")
    try:
        df = load_raw_data()
        print(f"  - {len(df)} lignes chargées")
    except FileNotFoundError as e:
        print(f"  - ERREUR: {e}")
        print("\n  Pour résoudre:")
        print("  1. Exécutez d'abord le notebook 01_data_exploration.ipynb")
        print("  2. Ou copiez le fichier data.csv dans data/raw/")
        return 1

    # 2. Préparer les transactions
    print("\n[2/4] Préparation des transactions...")
    try:
        transactions = prepare_transactions(df)
        transactions["order_purchase_timestamp"] = pd.to_datetime(
            transactions["order_purchase_timestamp"]
        )
        print(f"  - {len(transactions)} transactions")
        print(f"  - {transactions['customer_unique_id'].nunique()} clients uniques")
    except ValueError as e:
        print(f"  - ERREUR: {e}")
        return 1

    # 3. Calculer les features RFM
    print("\n[3/4] Calcul des features RFM...")
    reference_date = transactions["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    print(f"  - Date de référence: {reference_date}")

    calculator = RFMCalculator(reference_date=reference_date)
    rfm_df = calculator.fit_transform(transactions)

    print(f"  - {len(rfm_df)} clients avec features RFM")
    print(f"  - Statistiques:")
    print(calculator.get_statistics().round(2).to_string().replace("\n", "\n    "))

    # 4. Segmentation
    print("\n[4/4] Segmentation des clients...")
    segmenter = CustomerSegmenter(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)
    labels = segmenter.fit_predict(rfm_df)

    # Ajouter les segments au DataFrame
    rfm_df["segment"] = labels

    # Afficher la distribution
    print(f"  - Distribution des segments:")
    for segment_id in sorted(rfm_df["segment"].unique()):
        count = (rfm_df["segment"] == segment_id).sum()
        pct = count / len(rfm_df) * 100
        print(f"    Segment {segment_id}: {count:,} clients ({pct:.1f}%)")

    # Sauvegarder les données
    print("\n[5/5] Sauvegarde des fichiers...")

    # RFM avec segments (Parquet)
    rfm_path = PROCESSED_DATA_DIR / "customers_rfm.parquet"
    rfm_df.to_parquet(rfm_path)
    print(f"  - {rfm_path}")

    # RFM avec segments (CSV backup)
    rfm_csv_path = PROCESSED_DATA_DIR / "customers_rfm.csv"
    rfm_df.to_csv(rfm_csv_path)
    print(f"  - {rfm_csv_path}")

    # Modèle
    model_path, scaler_path = segmenter.save(MODELS_DIR)
    print(f"  - {model_path}")
    print(f"  - {scaler_path}")

    print("\n" + "=" * 60)
    print("Préparation terminée avec succès!")
    print("=" * 60)
    print("\nPour lancer le dashboard:")
    print("  streamlit run app/app.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
