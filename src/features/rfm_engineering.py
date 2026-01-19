"""
RFM (Recency, Frequency, Monetary) feature engineering module.

This module calculates RFM metrics for customer segmentation analysis.
RFM is a method for analyzing customer value based on:
- Recency: How recently a customer made a purchase
- Frequency: How often they purchase
- Monetary: How much they spend
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RFMCalculator:
    """
    Calculate RFM metrics for customer segmentation.

    Attributes:
        reference_date (datetime): The reference date for recency calculation.
    """

    def __init__(self, reference_date: Optional[datetime] = None):
        """
        Initialize RFM calculator.

        Parameters:
            reference_date (datetime, optional): Reference date for recency calculation.
                If None, uses the maximum date in the dataset.
        """
        self.reference_date = reference_date

    def calculate_rfm(
        self,
        data: pd.DataFrame,
        customer_id_col: str = 'customer_unique_id',
        date_col: str = 'order_purchase_timestamp',
        monetary_col: str = 'payment_value'
    ) -> pd.DataFrame:
        """
        Calculate RFM metrics from transaction data.

        Parameters:
            data (pd.DataFrame): Transaction-level data.
            customer_id_col (str): Column name for customer ID.
            date_col (str): Column name for transaction date.
            monetary_col (str): Column name for transaction value.

        Returns:
            pd.DataFrame: RFM metrics per customer.
        """
        logger.info("Calculating RFM metrics...")

        # Ensure date column is datetime
        df = data.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

        # Set reference date if not provided
        if self.reference_date is None:
            self.reference_date = df[date_col].max()
            logger.info(f"Reference date set to: {self.reference_date}")

        # Calculate RFM metrics
        rfm = df.groupby(customer_id_col).agg({
            date_col: lambda x: (self.reference_date - x.max()).days,  # Recency
            'order_id': 'nunique',  # Frequency (unique orders)
            monetary_col: 'sum'  # Monetary
        }).reset_index()

        # Rename columns
        rfm.columns = [customer_id_col, 'Recency', 'Frequency', 'Monetary']

        # Remove negative recency values (data quality issue)
        rfm = rfm[rfm['Recency'] >= 0]

        logger.info(f"RFM calculated for {len(rfm)} customers")
        logger.info(f"Recency range: {rfm['Recency'].min()}-{rfm['Recency'].max()} days")
        logger.info(f"Frequency range: {rfm['Frequency'].min()}-{rfm['Frequency'].max()} orders")
        logger.info(f"Monetary range: ${rfm['Monetary'].min():.2f}-${rfm['Monetary'].max():.2f}")

        return rfm

    def add_rfm_scores(
        self,
        rfm: pd.DataFrame,
        n_quantiles: int = 5
    ) -> pd.DataFrame:
        """
        Add RFM scores based on quantiles.

        Parameters:
            rfm (pd.DataFrame): RFM metrics dataframe.
            n_quantiles (int): Number of quantiles for scoring (default: 5).

        Returns:
            pd.DataFrame: RFM with score columns.
        """
        logger.info(f"Adding RFM scores ({n_quantiles} quantiles)...")

        df = rfm.copy()

        # Calculate scores (1 to n_quantiles)
        # Note: Recency is inverted (lower is better)
        df['R_Score'] = pd.qcut(df['Recency'], q=n_quantiles, labels=False, duplicates='drop')
        df['R_Score'] = n_quantiles - df['R_Score']  # Invert so recent = high score

        df['F_Score'] = pd.qcut(df['Frequency'], q=n_quantiles, labels=False, duplicates='drop') + 1
        df['M_Score'] = pd.qcut(df['Monetary'], q=n_quantiles, labels=False, duplicates='drop') + 1

        # Combined RFM score
        df['RFM_Score'] = df['R_Score'].astype(str) + df['F_Score'].astype(str) + df['M_Score'].astype(str)

        logger.info("RFM scores added")
        return df


def calculate_customer_metrics(
    data: pd.DataFrame,
    customer_id_col: str = 'customer_unique_id'
) -> pd.DataFrame:
    """
    Calculate additional customer-level metrics beyond RFM.

    Parameters:
        data (pd.DataFrame): Transaction data with reviews.
        customer_id_col (str): Column name for customer ID.

    Returns:
        pd.DataFrame: Customer-level aggregated metrics.
    """
    logger.info("Calculating additional customer metrics...")

    metrics = data.groupby(customer_id_col).agg({
        'order_id': 'nunique',
        'product_id': 'nunique',
        'payment_value': ['sum', 'mean', 'std'],
        'mean_review_score': 'mean',
        'freight_value': 'sum'
    }).reset_index()

    # Flatten column names
    metrics.columns = ['_'.join(col).strip('_') for col in metrics.columns.values]

    logger.info(f"Metrics calculated for {len(metrics)} customers")
    return metrics


def merge_rfm_with_reviews(
    rfm_data: pd.DataFrame,
    transaction_data: pd.DataFrame,
    customer_id_col: str = 'customer_unique_id'
) -> pd.DataFrame:
    """
    Merge RFM data with average review scores per customer.

    Parameters:
        rfm_data (pd.DataFrame): RFM metrics.
        transaction_data (pd.DataFrame): Transaction data with review scores.
        customer_id_col (str): Column name for customer ID.

    Returns:
        pd.DataFrame: RFM data with review scores.
    """
    logger.info("Merging RFM data with review scores...")

    # Calculate average review score per customer
    review_scores = transaction_data.groupby(customer_id_col).agg({
        'mean_review_score': 'mean'
    }).reset_index()

    # Merge
    rfm_with_reviews = rfm_data.merge(
        review_scores,
        on=customer_id_col,
        how='left'
    )

    logger.info(f"Merged data: {len(rfm_with_reviews)} rows")
    return rfm_with_reviews


if __name__ == "__main__":
    # Example usage
    from src.data.data_loader import load_processed_data

    # Load transaction data
    transactions = load_processed_data("data/processed/transactions_with_reviews.csv")

    # Calculate RFM
    rfm_calc = RFMCalculator()
    rfm = rfm_calc.calculate_rfm(
        transactions,
        customer_id_col='customer_unique_id',
        date_col='order_purchase_timestamp',
        monetary_col='payment_value'
    )

    # Add RFM scores
    rfm_scored = rfm_calc.add_rfm_scores(rfm, n_quantiles=5)

    # Merge with reviews
    rfm_final = merge_rfm_with_reviews(rfm, transactions)

    # Save
    from src.data.data_loader import save_processed_data
    save_processed_data(rfm_final, "data/processed/data_RFM.csv")
