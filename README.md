# Olist Customer Segmentation

[![CI](https://github.com/thomasmebarki/olist-customer-segmentation/actions/workflows/ci.yml/badge.svg)](https://github.com/thomasmebarki/olist-customer-segmentation/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/thomasmebarki/olist-customer-segmentation/branch/main/graph/badge.svg)](https://codecov.io/gh/thomasmebarki/olist-customer-segmentation)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: ruff-format](https://img.shields.io/badge/code%20style-ruff--format-black)](https://github.com/astral-sh/ruff)

Customer segmentation system for Olist e-commerce using RFM analysis and KMeans clustering. Includes an interactive Streamlit dashboard for exploring customer segments.

![Dashboard Preview](docs/dashboard_preview.png)

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Dashboard](#dashboard)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements a customer segmentation solution for **Olist**, a Brazilian e-commerce marketplace. Using the RFM (Recency, Frequency, Monetary) framework and unsupervised machine learning, we identify distinct customer segments to enable targeted marketing strategies.

### Business Context

- **Company**: Olist - Brazilian e-commerce platform
- **Challenge**: Segment 96,000+ customers for personalized marketing
- **Solution**: RFM-based clustering with automated maintenance recommendations

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

## Features

- **RFM Feature Engineering**: Automated calculation of Recency, Frequency, and Monetary values
- **KMeans Clustering**: Optimized segmentation with silhouette analysis
- **Interactive Dashboard**: Streamlit-based visualization of segments
- **Model Persistence**: Save/load trained models for production use
- **Maintenance Simulation**: ARI-based drift detection for model updates

## Installation

### Prerequisites

- Python 3.10 or higher
- Git LFS (for large data files)

### Quick Setup

```bash
# Clone and setup
git clone https://github.com/thomasmebarki/olist-customer-segmentation.git
cd olist-customer-segmentation

# Run the setup script
./scripts/setup.sh

# Or manually:
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

### Using Make

```bash
make install-dev  # Install with dev dependencies
make test         # Run tests
make lint         # Run linters
make serve        # Start dashboard
```

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

# Show project info
olist-segment info --verbose
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

# Or using Make
make serve
```

Then open http://localhost:8501 in your browser.

### Run Notebooks

The analysis is documented in Jupyter notebooks:

```bash
jupyter notebook notebooks/
```

| Notebook | Description |
|----------|-------------|
| `01_data_exploration.ipynb` | Exploratory data analysis |
| `02_feature_engineering.ipynb` | RFM feature creation |
| `03_modeling.ipynb` | Clustering and evaluation |
| `04_results_analysis.ipynb` | Segment analysis and maintenance |

### Run Tests

```bash
pytest tests/ -v --cov=src --cov-report=html
```

## Project Structure

```
olist-customer-segmentation/
├── .github/workflows/            # CI/CD pipelines
│   ├── ci.yml                    # Continuous Integration
│   ├── release.yml               # Release automation
│   └── pr-preview.yml            # PR preview comments
├── app/                          # Streamlit dashboard
│   ├── app.py                    # Main entry point
│   └── pages/                    # Dashboard pages
├── data/
│   ├── raw/                      # Original datasets
│   └── processed/                # Processed RFM data
├── docs/                         # Documentation
├── models/                       # Saved models
├── notebooks/                    # Jupyter notebooks
├── scripts/                      # Automation scripts
│   ├── setup.sh                  # Environment setup
│   ├── train.sh                  # Training pipeline
│   └── download_data.sh          # Data download
├── src/                          # Source code
│   ├── cli.py                    # CLI commands
│   ├── config.py                 # Configuration
│   ├── data/                     # Data loading & preprocessing
│   ├── features/                 # Feature engineering (RFM)
│   ├── models/                   # Clustering & evaluation
│   └── visualization/            # Plotting functions
├── tests/                        # Unit tests (152 tests)
├── Makefile                      # Build automation
├── pyproject.toml                # Project configuration
├── CHANGELOG.md                  # Version history
└── README.md
```

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

## Dashboard

The interactive Streamlit dashboard provides:

- **Overview**: Key metrics and segment distribution
- **Segments**: Detailed analysis of each customer segment
- **Explorer**: Interactive 3D visualization
- **About**: Project documentation

### Screenshots

| Overview | Segment Analysis |
|----------|------------------|
| ![Overview](docs/screenshot_overview.png) | ![Segments](docs/screenshot_segments.png) |

## Tech Stack

- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn
- **Visualization**: matplotlib, seaborn, plotly
- **Dashboard**: Streamlit
- **Testing**: pytest
- **Code Quality**: black, ruff

## Data Source

This project uses the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) from Kaggle.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Run tests (`make check`)
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

### Development Setup

```bash
# Install dev dependencies
make install-dev

# Run all checks
make check

# Format code
make format
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Thomas Mebarki**

- GitHub: [@thomasmebarki](https://github.com/thomasmebarki)

---

*This project was developed as part of a Data Science training program, focusing on unsupervised learning and customer segmentation techniques.*
