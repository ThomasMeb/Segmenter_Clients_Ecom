# Back Market - Customer Segmentation

[![CI](https://github.com/ThomasMeb/backmarket-segmentation/actions/workflows/ci.yml/badge.svg)](https://github.com/ThomasMeb/backmarket-segmentation/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Customer segmentation system using RFM analysis and KMeans clustering. Includes an interactive Streamlit dashboard for exploring customer segments.

> **Portfolio Notice** : Ce repository est une **version portfolio** d'une mission freelance réalisée pour **Back Market** entre décembre 2023 et février 2024. Pour des raisons de confidentialité, les données client ont été remplacées par le dataset public [Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), qui présente une structure similaire. La méthodologie RFM et l'architecture de clustering sont identiques à celles déployées en production.

![Dashboard Preview](docs/dashboard_preview.png)

---

## Mission Context

| Attribute | Details |
|-----------|---------|
| **Client** | Back Market |
| **Mission Type** | Freelance - Data Science |
| **Period** | December 2023 - February 2024 |
| **Role** | Data Scientist |

### About Back Market

**Back Market** est la licorne française leader du reconditionnement électronique. L'entreprise souhaitait segmenter sa base clients pour personnaliser ses campagnes marketing et améliorer la rétention.

### Deliverables

- Pipeline ML complet (train, predict, evaluate)
- Dashboard Streamlit interactif
- CLI documentée
- Simulation de maintenance (ARI drift detection)

---

## Key Results

| Metric | Value |
|--------|-------|
| **Customers Analyzed** | 96,096 |
| **Segments Identified** | 4 |
| **Silhouette Score** | 0.68 |
| **Model Update Frequency** | Every 3-4 months |

### Customer Segments

| Segment | % of Customers | Characteristics | Recommended Action |
|---------|---------------|-----------------|-------------------|
| **Recent** | 54% | Recent purchase, low frequency | Convert to loyal |
| **Loyal** | 3% | Regular purchases | Retention program |
| **Dormant** | 40% | Inactive for months | Reactivation campaign |
| **VIP** | 3% | High value customers | Premium treatment |

---

## Features

- **RFM Feature Engineering**: Automated calculation of Recency, Frequency, and Monetary values
- **KMeans Clustering**: Optimized segmentation with silhouette analysis
- **Interactive Dashboard**: Streamlit-based visualization of segments
- **Model Persistence**: Save/load trained models for production use
- **Maintenance Simulation**: ARI-based drift detection for model updates

---

## Installation

### Prerequisites

- Python 3.10 or higher
- Git LFS (for large data files)

### Quick Setup

```bash
# Clone and setup
git clone https://github.com/ThomasMeb/backmarket-segmentation.git
cd backmarket-segmentation

# Run the setup script
./scripts/setup.sh

# Or manually:
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

### Data Setup

Les données ne sont pas incluses dans le repository. Téléchargez-les depuis Kaggle :

```bash
# 1. Configurer l'API Kaggle (une seule fois)
pip install kaggle
# Télécharger kaggle.json depuis https://www.kaggle.com/settings
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 2. Télécharger les données (~100MB)
./scripts/download_data.sh
```

Ou téléchargez manuellement depuis [Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

### Using Make

```bash
make install-dev  # Install with dev dependencies
make test         # Run tests
make lint         # Run linters
make serve        # Start dashboard
```

---

## Usage

### CLI Commands

```bash
# Train the model
olist-segment train --input data/raw/data.csv --verbose

# Predict segments for new customers
olist-segment predict --input new_customers.csv --output predictions.csv

# Evaluate model metrics
olist-segment evaluate --verbose

# Start the dashboard
olist-segment serve --port 8501
```

### Python API

```python
from src.data.loader import load_transactions
from src.features.rfm import RFMCalculator
from src.models.clustering import CustomerSegmenter

# Load data
df = load_transactions("data/raw/data.csv")

# Calculate RFM features
calculator = RFMCalculator(reference_date="2018-09-01")
rfm = calculator.fit_transform(df)

# Segment customers
segmenter = CustomerSegmenter(n_clusters=4)
labels = segmenter.fit_predict(rfm)

# Get segment summary
summary = segmenter.get_segment_summary(rfm)
print(summary)
```

### Run the Dashboard

```bash
# Using CLI
olist-segment serve

# Or directly with Streamlit
streamlit run app/app.py
```

Then open http://localhost:8501 in your browser.

---

## Project Structure

```
backmarket-segmentation/
├── .github/workflows/            # CI/CD pipelines
├── app/                          # Streamlit dashboard
│   ├── app.py                    # Main entry point
│   └── pages/                    # Dashboard pages
├── data/
│   ├── raw/                      # Original datasets (Olist for portfolio)
│   └── processed/                # Processed RFM data
├── docs/                         # Documentation
├── models/                       # Saved models
├── notebooks/                    # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_results_analysis.ipynb
├── src/                          # Source code
│   ├── cli.py                    # CLI commands
│   ├── data/                     # Data loading & preprocessing
│   ├── features/                 # Feature engineering (RFM)
│   ├── models/                   # Clustering & evaluation
│   └── visualization/            # Plotting functions
├── tests/                        # Unit tests (152 tests)
├── Makefile                      # Build automation
└── pyproject.toml                # Project configuration
```

---

## Methodology

### 1. RFM Analysis

The RFM model evaluates customers based on:

- **Recency (R)**: Days since last purchase
- **Frequency (F)**: Total number of orders
- **Monetary (M)**: Total amount spent

### 2. Clustering Algorithm

We use **KMeans** with the following configuration:

```python
KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10,
    max_iter=300
)
```

### 3. Model Selection

The optimal number of clusters (k=4) was determined using:
- Elbow method (inertia analysis)
- Silhouette score optimization
- Business interpretability

### 4. Model Maintenance

Using Adjusted Rand Index (ARI) simulation:
- **Recommended update frequency**: Every 3-4 months
- **Alert threshold**: ARI < 0.8

---

## Tech Stack

- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn
- **Visualization**: matplotlib, seaborn, plotly
- **Dashboard**: Streamlit
- **Testing**: pytest (152 tests)
- **Code Quality**: ruff

---

## Data Source

This portfolio version uses the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) from Kaggle. The original Back Market data remains confidential.

---

## Testing

```bash
pytest tests/ -v --cov=src --cov-report=html
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Author

**Thomas Mebarki**
Data Scientist & ML Engineer

- GitHub: [@ThomasMeb](https://github.com/ThomasMeb)
- LinkedIn: [Thomas Mebarki](https://linkedin.com/in/thomas-mebarki)

---

*Mission réalisée pour Back Market entre décembre 2023 et février 2024. Version portfolio adaptée sur le dataset Olist.*
