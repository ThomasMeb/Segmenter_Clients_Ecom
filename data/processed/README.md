# Processed Data

This directory contains preprocessed and feature-engineered datasets derived from the raw Olist data.

## Files

| File | Rows | Description | Generation Script |
|------|------|-------------|-------------------|
| `data.csv` | 98,666 | Merged transaction-level data with customer and order info | `src/data/data_loader.py` |
| `data_RFM.csv` | 95,420 | **RFM features** (Recency, Frequency, Monetary) per customer | `src/features/rfm_engineering.py` |
| `data_2.csv` | 95,420 | RFM + mean review score (customer satisfaction) | `src/features/rfm_engineering.py` |
| `rfm_clustered.csv` | 95,420 | RFM data with cluster assignments from trained model | `train.py` |
| `predictions.csv` | Variable | Customer predictions with cluster labels | `predict.py` |

## Feature Descriptions

### RFM Metrics

- **Recency (R):** Days since customer's last purchase (0-728 days)
- **Frequency (F):** Number of orders placed by customer (1-16)
- **Monetary (M):** Total amount spent by customer ($0.85 - $7,388)

### Additional Features

- **mean_review_score:** Average satisfaction score across all customer reviews (1-5 scale)
- **Cluster:** Assigned segment (0-3) from K-Means clustering model
- **Cluster_Description:** Human-readable cluster name

## Data Processing Pipeline

```
Raw Olist Data (9 CSVs)
    ↓
[data.csv] - Merge orders + customers + items + payments
    ↓
[data_RFM.csv] - Calculate Recency, Frequency, Monetary
    ↓
[data_2.csv] - Add mean_review_score
    ↓
[rfm_clustered.csv] - Apply K-Means clustering (k=4)
    ↓
[predictions.csv] - Predict new customer segments
```

## Usage

These files can be loaded directly using:

```python
from src.data.data_loader import load_processed_data

# Load RFM data
rfm_data = load_processed_data("data/processed/data_RFM.csv")

# Load clustered results
clustered_data = load_processed_data("data/processed/rfm_clustered.csv")
```

## Notes

- CSV files are gitignored (regenerated from raw data)
- Customer IDs are hashed for privacy
- Missing values handled via IterativeImputer
- Negative recency values removed (data quality issue)
