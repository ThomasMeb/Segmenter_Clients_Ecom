"""
Module de calcul des features RFM.

Ce module implémente le calcul des features Recency, Frequency, Monetary
pour la segmentation client.
"""

from datetime import datetime
from typing import Optional, Union

import pandas as pd
import numpy as np

from src.config import (
    CUSTOMER_ID_COL,
    ORDER_ID_COL,
    DATE_COL,
    AMOUNT_COL,
)


class RFMCalculator:
    """
    Calculateur de features RFM (Recency, Frequency, Monetary).

    Le modèle RFM est une technique de segmentation client basée sur
    le comportement d'achat :
    - Recency : Combien de temps depuis le dernier achat ?
    - Frequency : Combien d'achats au total ?
    - Monetary : Combien dépensé au total ?

    Attributes
    ----------
    reference_date : datetime
        Date de référence pour le calcul de la récence.
    rfm_data : pd.DataFrame
        DataFrame contenant les features RFM calculées.

    Examples
    --------
    >>> calculator = RFMCalculator(reference_date=datetime(2018, 9, 1))
    >>> rfm_df = calculator.fit_transform(transactions_df)
    >>> rfm_df.head()
                                  recency  frequency  monetary
    customer_unique_id
    7c396fd4830fd04220f754e42...      335          2     65.38
    """

    def __init__(self, reference_date: Optional[Union[datetime, str]] = None):
        """
        Initialise le calculateur RFM.

        Parameters
        ----------
        reference_date : datetime or str, optional
            Date de référence pour le calcul de la récence.
            Si non spécifié, utilise la date maximale des données.
        """
        if isinstance(reference_date, str):
            reference_date = pd.to_datetime(reference_date)
        self.reference_date = reference_date
        self.rfm_data: Optional[pd.DataFrame] = None

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les features RFM à partir des transactions.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame des transactions avec les colonnes :
            - customer_unique_id : identifiant client
            - order_purchase_timestamp : date de commande
            - price : montant de la commande

        Returns
        -------
        pd.DataFrame
            DataFrame avec les features RFM indexé par customer_unique_id.
        """
        df = df.copy()

        # Conversion de la date si nécessaire
        if df[DATE_COL].dtype == "object":
            df[DATE_COL] = pd.to_datetime(df[DATE_COL])

        # Date de référence
        if self.reference_date is None:
            self.reference_date = df[DATE_COL].max()

        # Calcul RFM
        rfm = df.groupby(CUSTOMER_ID_COL).agg({
            DATE_COL: "max",           # Date du dernier achat
            ORDER_ID_COL: "nunique",   # Nombre de commandes uniques
            AMOUNT_COL: "sum",         # Montant total
        })

        # Renommage des colonnes
        rfm.columns = ["last_purchase", "frequency", "monetary"]

        # Calcul de la récence (en jours)
        rfm["recency"] = (self.reference_date - rfm["last_purchase"]).dt.days

        # Sélection et réordonnancement des colonnes
        self.rfm_data = rfm[["recency", "frequency", "monetary"]]

        return self.rfm_data

    def get_statistics(self) -> pd.DataFrame:
        """
        Retourne les statistiques descriptives des features RFM.

        Returns
        -------
        pd.DataFrame
            Statistiques descriptives (mean, std, min, max, quartiles).

        Raises
        ------
        ValueError
            Si fit_transform n'a pas été appelé.
        """
        if self.rfm_data is None:
            raise ValueError("Appelez fit_transform() d'abord.")

        return self.rfm_data.describe()

    def save(self, filepath: str, format: str = "parquet") -> None:
        """
        Sauvegarde les données RFM.

        Parameters
        ----------
        filepath : str
            Chemin du fichier de sortie.
        format : str, default "parquet"
            Format de sortie : "parquet" ou "csv".
        """
        if self.rfm_data is None:
            raise ValueError("Appelez fit_transform() d'abord.")

        if format == "parquet":
            self.rfm_data.to_parquet(filepath)
        elif format == "csv":
            self.rfm_data.to_csv(filepath)
        else:
            raise ValueError(f"Format non supporté : {format}")


def calculate_rfm(
    df: pd.DataFrame,
    customer_col: str = CUSTOMER_ID_COL,
    date_col: str = DATE_COL,
    amount_col: str = AMOUNT_COL,
    reference_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Fonction utilitaire pour calculer les features RFM.

    Cette fonction est un raccourci pour utiliser RFMCalculator
    sans instancier la classe.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame des transactions.
    customer_col : str, default "customer_unique_id"
        Nom de la colonne identifiant client.
    date_col : str, default "order_purchase_timestamp"
        Nom de la colonne date.
    amount_col : str, default "price"
        Nom de la colonne montant.
    reference_date : datetime, optional
        Date de référence pour la récence.

    Returns
    -------
    pd.DataFrame
        DataFrame avec les features RFM.

    Examples
    --------
    >>> rfm = calculate_rfm(transactions, reference_date=datetime(2018, 9, 1))
    >>> rfm.head()
    """
    # Renommage temporaire des colonnes si nécessaire
    df = df.rename(columns={
        customer_col: CUSTOMER_ID_COL,
        date_col: DATE_COL,
        amount_col: AMOUNT_COL,
    })

    calculator = RFMCalculator(reference_date=reference_date)
    return calculator.fit_transform(df)
