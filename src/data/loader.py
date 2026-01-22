"""
Module de chargement des données.

Ce module fournit des fonctions pour charger et valider les données
depuis différentes sources (CSV, Parquet).
"""

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from src.config import (
    RAW_TRANSACTIONS_FILE,
    PROCESSED_RFM_FILE,
    CUSTOMER_ID_COL,
    ORDER_ID_COL,
    DATE_COL,
    AMOUNT_COL,
)


def load_transactions(
    filepath: Optional[Union[str, Path]] = None,
    parse_dates: bool = True,
) -> pd.DataFrame:
    """
    Charge les données de transactions depuis un fichier CSV.

    Parameters
    ----------
    filepath : str or Path, optional
        Chemin vers le fichier CSV. Par défaut, utilise le fichier
        configuré dans config.py.
    parse_dates : bool, default True
        Si True, convertit les colonnes de dates en datetime.

    Returns
    -------
    pd.DataFrame
        DataFrame contenant les transactions avec les colonnes :
        - customer_unique_id : identifiant unique du client
        - order_id : identifiant de la commande
        - price : montant de la transaction
        - order_purchase_timestamp : date de la commande

    Raises
    ------
    FileNotFoundError
        Si le fichier spécifié n'existe pas.
    ValueError
        Si les colonnes requises sont manquantes.

    Examples
    --------
    >>> df = load_transactions()
    >>> df.head()
       customer_unique_id  order_id  price  order_purchase_timestamp
    0  7c396fd4830fd042...  e481f51...  29.99       2017-10-02 10:56:33
    """
    if filepath is None:
        filepath = RAW_TRANSACTIONS_FILE

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Le fichier n'existe pas : {filepath}")

    # Colonnes de dates à parser
    date_columns = [DATE_COL] if parse_dates else None

    df = pd.read_csv(filepath, parse_dates=date_columns)

    # Suppression de la colonne Unnamed si présente
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Validation des colonnes requises
    required_columns = [CUSTOMER_ID_COL, ORDER_ID_COL, AMOUNT_COL]
    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Colonnes manquantes : {missing_columns}")

    return df


def load_rfm_data(
    filepath: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Charge les données RFM pré-calculées.

    Parameters
    ----------
    filepath : str or Path, optional
        Chemin vers le fichier Parquet ou CSV. Par défaut, utilise
        le fichier configuré dans config.py.

    Returns
    -------
    pd.DataFrame
        DataFrame contenant les features RFM :
        - customer_unique_id : identifiant unique du client (index)
        - recency : jours depuis le dernier achat
        - frequency : nombre de commandes
        - monetary : montant total dépensé

    Raises
    ------
    FileNotFoundError
        Si le fichier spécifié n'existe pas.

    Examples
    --------
    >>> rfm = load_rfm_data()
    >>> rfm.head()
                                  recency  frequency  monetary
    customer_unique_id
    7c396fd4830fd04220f754e42...      335          2     65.38
    """
    if filepath is None:
        filepath = PROCESSED_RFM_FILE

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Le fichier n'existe pas : {filepath}")

    # Chargement selon le format
    if filepath.suffix == ".parquet":
        df = pd.read_parquet(filepath)
    elif filepath.suffix == ".csv":
        df = pd.read_csv(filepath, index_col=0)
    else:
        raise ValueError(f"Format non supporté : {filepath.suffix}")

    return df


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list[str],
    name: str = "DataFrame",
) -> bool:
    """
    Valide qu'un DataFrame contient les colonnes requises.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à valider.
    required_columns : list of str
        Liste des colonnes requises.
    name : str, default "DataFrame"
        Nom du DataFrame pour les messages d'erreur.

    Returns
    -------
    bool
        True si la validation réussit.

    Raises
    ------
    ValueError
        Si des colonnes sont manquantes.
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"{name} : colonnes manquantes {missing}")
    return True
