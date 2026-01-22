"""
Module de prétraitement des données.

Ce module fournit des fonctions pour nettoyer et transformer
les données brutes avant l'analyse.
"""

from typing import Optional

import pandas as pd
import numpy as np

from src.config import (
    CUSTOMER_ID_COL,
    ORDER_ID_COL,
    DATE_COL,
    AMOUNT_COL,
)


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les données de transactions.

    Opérations effectuées :
    - Suppression des doublons
    - Suppression des valeurs manquantes
    - Suppression des montants négatifs ou nuls
    - Conversion des types de données

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame brut des transactions.

    Returns
    -------
    pd.DataFrame
        DataFrame nettoyé.

    Examples
    --------
    >>> df_clean = clean_transactions(df_raw)
    >>> df_clean.isnull().sum().sum()
    0
    """
    df = df.copy()

    # Suppression des doublons
    initial_rows = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)

    if duplicates_removed > 0:
        print(f"Doublons supprimés : {duplicates_removed}")

    # Suppression des valeurs manquantes dans les colonnes critiques
    critical_columns = [CUSTOMER_ID_COL, ORDER_ID_COL, AMOUNT_COL]
    df = df.dropna(subset=critical_columns)

    # Suppression des montants négatifs ou nuls
    df = df[df[AMOUNT_COL] > 0]

    # Conversion de la date si nécessaire
    if DATE_COL in df.columns and df[DATE_COL].dtype == "object":
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    return df.reset_index(drop=True)


def prepare_for_rfm(
    df: pd.DataFrame,
    reference_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Prépare les données pour le calcul RFM.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame des transactions nettoyé.
    reference_date : pd.Timestamp, optional
        Date de référence pour le calcul de la récence.
        Par défaut, utilise la date maximale des données.

    Returns
    -------
    pd.DataFrame
        DataFrame avec les colonnes nécessaires au calcul RFM.

    Examples
    --------
    >>> df_prep = prepare_for_rfm(df_clean)
    >>> df_prep.columns.tolist()
    ['customer_unique_id', 'order_purchase_timestamp', 'price']
    """
    df = df.copy()

    # Sélection des colonnes nécessaires
    required_cols = [CUSTOMER_ID_COL, DATE_COL, AMOUNT_COL]
    df = df[required_cols]

    # Conversion de la date si nécessaire
    if df[DATE_COL].dtype == "object":
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    # Tri par date
    df = df.sort_values(DATE_COL)

    return df


def remove_outliers(
    df: pd.DataFrame,
    column: str,
    method: str = "iqr",
    threshold: float = 1.5,
) -> pd.DataFrame:
    """
    Supprime les outliers d'une colonne.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame source.
    column : str
        Nom de la colonne à traiter.
    method : str, default "iqr"
        Méthode de détection : "iqr" ou "zscore".
    threshold : float, default 1.5
        Seuil pour la détection (1.5 pour IQR, 3 pour zscore).

    Returns
    -------
    pd.DataFrame
        DataFrame sans les outliers.

    Examples
    --------
    >>> df_no_outliers = remove_outliers(df, "monetary", method="iqr")
    """
    df = df.copy()

    if method == "iqr":
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    elif method == "zscore":
        mean = df[column].mean()
        std = df[column].std()
        df = df[np.abs((df[column] - mean) / std) <= threshold]

    else:
        raise ValueError(f"Méthode inconnue : {method}")

    return df
