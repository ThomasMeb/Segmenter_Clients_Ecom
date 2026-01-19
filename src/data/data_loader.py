"""
Data loading utilities for Olist E-Commerce dataset.

This module provides functions to load and preprocess the Olist dataset,
including orders, customers, products, and reviews.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OlistDataLoader:
    """
    Loads and manages Olist E-Commerce datasets.

    Attributes:
        data_path (Path): Path to the directory containing raw data files.
    """

    def __init__(self, data_path: str = "data/raw"):
        """
        Initialize the data loader.

        Parameters:
            data_path (str): Path to directory containing Olist CSV files.
        """
        self.data_path = Path(data_path)
        self.datasets = {}

    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all Olist datasets into memory.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary of dataset names and dataframes.
        """
        dataset_files = {
            'orders': 'olist_orders_dataset.csv',
            'order_items': 'olist_order_items_dataset.csv',
            'order_reviews': 'olist_order_reviews_dataset.csv',
            'customers': 'olist_customers_dataset.csv',
            'order_payments': 'olist_order_payments_dataset.csv',
            'products': 'olist_products_dataset.csv',
            'sellers': 'olist_sellers_dataset.csv',
            'geolocation': 'olist_geolocation_dataset.csv',
            'product_category': 'product_category_name_translation.csv'
        }

        for name, filename in dataset_files.items():
            filepath = self.data_path / filename
            if filepath.exists():
                logger.info(f"Loading {name} dataset...")
                self.datasets[name] = pd.read_csv(filepath)
                logger.info(f"{name}: {len(self.datasets[name])} rows loaded")
            else:
                logger.warning(f"File not found: {filepath}")

        return self.datasets

    def load_specific_dataset(self, dataset_name: str) -> pd.DataFrame:
        """
        Load a specific dataset by name.

        Parameters:
            dataset_name (str): Name of the dataset (e.g., 'orders', 'customers').

        Returns:
            pd.DataFrame: The loaded dataset.
        """
        if dataset_name in self.datasets:
            return self.datasets[dataset_name]
        else:
            raise ValueError(f"Dataset '{dataset_name}' not loaded. Call load_all_datasets() first.")

    def merge_transaction_data(self) -> pd.DataFrame:
        """
        Merge orders, items, customers, and payments into a transaction-level dataset.

        Returns:
            pd.DataFrame: Merged transaction data.
        """
        logger.info("Merging transaction data...")

        # Start with orders
        df = self.datasets['orders'].copy()

        # Merge with customers
        df = df.merge(
            self.datasets['customers'],
            on='customer_id',
            how='left'
        )

        # Merge with order items
        df = df.merge(
            self.datasets['order_items'],
            on='order_id',
            how='left'
        )

        # Merge with order payments
        payments_agg = self.datasets['order_payments'].groupby('order_id').agg({
            'payment_value': 'sum',
            'payment_type': 'first'
        }).reset_index()

        df = df.merge(
            payments_agg,
            on='order_id',
            how='left'
        )

        # Convert date columns to datetime
        date_columns = [col for col in df.columns if 'date' in col or 'timestamp' in col]
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        logger.info(f"Transaction data merged: {len(df)} rows")
        return df

    def merge_with_reviews(self, transaction_data: pd.DataFrame) -> pd.DataFrame:
        """
        Merge transaction data with customer reviews.

        Parameters:
            transaction_data (pd.DataFrame): Transaction-level data.

        Returns:
            pd.DataFrame: Data with review scores.
        """
        logger.info("Merging with review data...")

        # Aggregate reviews by order
        reviews_agg = self.datasets['order_reviews'].groupby('order_id').agg({
            'review_score': 'mean'
        }).reset_index()
        reviews_agg.rename(columns={'review_score': 'mean_review_score'}, inplace=True)

        # Merge
        df = transaction_data.merge(
            reviews_agg,
            on='order_id',
            how='left'
        )

        logger.info(f"Data with reviews: {len(df)} rows")
        return df


def load_processed_data(filepath: str) -> pd.DataFrame:
    """
    Load a preprocessed dataset (e.g., data_RFM.csv, data_2.csv).

    Parameters:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    logger.info(f"Loading processed data from {filepath}...")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def save_processed_data(df: pd.DataFrame, filepath: str) -> None:
    """
    Save a processed dataset to CSV.

    Parameters:
        df (pd.DataFrame): Dataframe to save.
        filepath (str): Destination file path.
    """
    logger.info(f"Saving data to {filepath}...")
    df.to_csv(filepath, index=False)
    logger.info(f"Saved {len(df)} rows to {filepath}")


if __name__ == "__main__":
    # Example usage
    loader = OlistDataLoader(data_path="data/raw")
    datasets = loader.load_all_datasets()

    # Create transaction data
    transaction_data = loader.merge_transaction_data()
    save_processed_data(transaction_data, "data/processed/transactions.csv")

    # Merge with reviews
    data_with_reviews = loader.merge_with_reviews(transaction_data)
    save_processed_data(data_with_reviews, "data/processed/transactions_with_reviews.csv")
