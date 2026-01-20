# Usage Guide

Complete guide for using the Customer Segmentation system.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Data Preparation](#data-preparation)
3. [Training a Model](#training-a-model)
4. [Making Predictions](#making-predictions)
5. [Evaluating Performance](#evaluating-performance)
6. [Using the Dashboard](#using-the-dashboard)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Minimal Example

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place Olist data in data/raw/

# 3. Train the model
python train.py

# 4. Launch dashboard
streamlit run app.py
```

---

## Data Preparation

### Step 1: Download Olist Dataset

1. Visit [Kaggle Olist Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
2. Download and extract all CSV files
3. Place files in `data/raw/`:
   - olist_orders_dataset.csv
   - olist_order_items_dataset.csv
   - olist_order_reviews_dataset.csv
   - olist_customers_dataset.csv
   - olist_order_payments_dataset.csv
   - olist_products_dataset.csv
   - olist_sellers_dataset.csv
   - olist_geolocation_dataset.csv
   - product_category_name_translation.csv

### Step 2: Process Raw Data (Optional)

The training script automatically processes raw data, but you can do it manually:

```python
from src.data.data_loader import OlistDataLoader
from src.features.rfm_engineering import RFMCalculator

# Load raw data
loader = OlistDataLoader(data_path="data/raw")
datasets = loader.load_all_datasets()

# Create transaction data
transaction_data = loader.merge_transaction_data()
data_with_reviews = loader.merge_with_reviews(transaction_data)

# Calculate RFM
rfm_calc = RFMCalculator()
rfm_data = rfm_calc.calculate_rfm(data_with_reviews)

# Save
from src.data.data_loader import save_processed_data
save_processed_data(rfm_data, "data/processed/data_RFM.csv")
```

---

## Training a Model

### Basic Training

```bash
python train.py
```

This will:
1. Load RFM data from `data/processed/data_RFM.csv`
2. Train K-Means clustering (k=4, configured in `config.yaml`)
3. Save model to `outputs/models/customer_segmentation_model.pkl`
4. Generate visualizations in `outputs/figures/`
5. Save clustered data to `data/processed/rfm_clustered.csv`

### Find Optimal K

Use the elbow method to determine the best number of clusters:

```bash
python train.py --find-optimal-k
```

This tests k from 2 to 15 and generates an elbow curve plot.

### Custom Configuration

Create a custom config file:

```yaml
# my_config.yaml
clustering:
  algorithm: "kmeans"
  kmeans:
    n_clusters: 5  # Try 5 clusters instead of 4
    random_state: 42
```

Train with custom config:

```bash
python train.py --config my_config.yaml
```

### Using Different Algorithms

Edit `config.yaml` to try DBSCAN or Agglomerative clustering:

```yaml
clustering:
  algorithm: "dbscan"  # or "agglomerative"
  dbscan:
    eps: 0.5
    min_samples: 10
```

---

## Making Predictions

### Predict on New Customers

```bash
# Prepare new customer data (CSV with Recency, Frequency, Monetary columns)
python predict.py --input data/new_customers.csv --output data/predictions.csv
```

### Programmatic Prediction

```python
from src.models.clustering import CustomerSegmenter
from src.data.data_loader import load_processed_data
import pandas as pd

# Load trained model
model = CustomerSegmenter.load_model("outputs/models/customer_segmentation_model.pkl")

# Load new customer data
new_customers = pd.DataFrame({
    'customer_unique_id': ['cust_001', 'cust_002'],
    'Recency': [50, 300],
    'Frequency': [3, 1],
    'Monetary': [450, 120]
})

# Predict clusters
labels = model.predict(new_customers)
new_customers['Cluster'] = labels

print(new_customers)
```

---

## Evaluating Performance

### Generate Evaluation Report

```bash
python evaluate.py
```

This generates:
- Evaluation metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin)
- Cluster profile visualizations
- Silhouette analysis plots
- Report saved to `outputs/figures/evaluation_report.yaml`

### Interpreting Metrics

**Silhouette Score** (range: -1 to 1)
- > 0.7: Excellent separation
- 0.5-0.7: Good separation
- 0.25-0.5: Weak separation
- < 0.25: Poor clustering

**Calinski-Harabasz Score** (higher is better)
- Measures ratio of between-cluster to within-cluster variance
- No fixed threshold, compare across models

**Davies-Bouldin Score** (lower is better)
- < 1: Good clustering
- 1-2: Acceptable
- > 2: Poor clustering

---

## Using the Dashboard

### Launch Dashboard

```bash
streamlit run app.py
```

Open browser at `http://localhost:8501`

### Dashboard Features

**Overview Tab**
- Total customer count
- Cluster distribution pie chart
- Average customer value metrics

**Cluster Analysis Tab**
- Interactive 3D RFM visualization
- Cluster profiles with expandable details
- Mean RFM values per cluster

**Customer Explorer Tab**
- Search customers by ID
- View top N customers by spend
- Filter by cluster

**RFM Analysis Tab**
- Recency, Frequency, Monetary distributions
- 2D scatter plots with customizable axes
- Marginal box plots

**Export Data Tab**
- Select clusters to export
- Download as CSV
- Preview data before download

---

## Configuration

### config.yaml Structure

```yaml
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"

clustering:
  algorithm: "kmeans"
  kmeans:
    n_clusters: 4
    random_state: 42
  feature_columns:
    - "Recency"
    - "Frequency"
    - "Monetary"

model:
  save_dir: "outputs/models"
  model_filename: "customer_segmentation_model.pkl"

visualization:
  save_dir: "outputs/figures"
  dpi: 300

cluster_descriptions:
  0: "Recent Browsers"
  1: "Loyal Repeat Buyers"
  2: "Dormant/Inactive"
  3: "VIP/High-Value"
```

### Customizing Cluster Descriptions

Edit the `cluster_descriptions` section to match your business context:

```yaml
cluster_descriptions:
  0: "New Customers - Onboarding Needed"
  1: "Champions - Reward & Retain"
  2: "At-Risk - Win Back Campaign"
  3: "VIPs - White Glove Service"
```

---

## Troubleshooting

### Common Issues

**Issue: "Model not found"**
```
Solution: Run `python train.py` first to generate the model
```

**Issue: "Clustered data not found"**
```
Solution: Training must complete successfully before evaluation/prediction
```

**Issue: "ValueError: could not convert string to float"**
```
Solution: Ensure RFM columns (Recency, Frequency, Monetary) are numeric
```

**Issue: Dashboard shows "No data"**
```
Solution: Check that data/processed/rfm_clustered.csv exists
```

**Issue: "ImportError: No module named 'src'"**
```
Solution: Ensure you're running from the project root directory
```

### Getting Help

1. Check this documentation
2. Review notebook examples in `notebooks/`
3. Examine module docstrings: `help(CustomerSegmenter)`
4. Open a GitHub issue with error logs

---

## Advanced Usage

### Temporal Stability Analysis

Monitor how cluster assignments drift over time:

```python
from src.evaluation.metrics import temporal_stability_analysis

stability_df = temporal_stability_analysis(
    data=time_series_rfm_data,
    model=trained_model,
    feature_columns=['Recency', 'Frequency', 'Monetary'],
    date_column='order_purchase_timestamp',
    n_weeks=52
)

# Plot ARI over time
from src.visualization.plots import plot_temporal_stability
plot_temporal_stability(stability_df, save_path='stability.png')
```

### Batch Processing

Process multiple customer files in batch:

```python
import glob

for filepath in glob.glob("data/new_customers/*.csv"):
    output_path = filepath.replace("new_customers", "predictions")
    # Run prediction
    !python predict.py --input {filepath} --output {output_path}
```

---

## Next Steps

- Explore notebooks for detailed analysis examples
- Customize cluster descriptions for your business
- Integrate with CRM systems via predictions
- Schedule periodic model retraining (every 12-16 weeks recommended)
