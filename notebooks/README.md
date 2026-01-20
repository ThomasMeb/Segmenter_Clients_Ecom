# Jupyter Notebooks

This directory contains exploratory data analysis (EDA) and experimental notebooks documenting the research and development process.

## Notebooks

| Notebook | Description | Key Insights |
|----------|-------------|--------------|
| `01_data_exploration.ipynb` | **Exploratory Data Analysis** | - Dataset structure and quality<br>- RFM distribution analysis<br>- Feature engineering exploration<br>- Customer behavior patterns |
| `02_clustering_experiments.ipynb` | **Algorithm Comparison** | - K-Means (k=2 to 15)<br>- DBSCAN hyperparameter tuning<br>- Agglomerative clustering<br>- Silhouette & Calinski-Harabasz scores<br>- **Winner: K-Means with k=4** |
| `03_temporal_stability_simulation.ipynb` | **Model Maintenance Analysis** | - Clustering stability over 52 weeks<br>- Adjusted Rand Index (ARI) tracking<br>- Retraining frequency recommendation<br>- Model drift detection |

## Running the Notebooks

### Prerequisites

```bash
# Install Jupyter dependencies
pip install jupyter notebook ipykernel

# Or use the full requirements
pip install -r requirements.txt
```

### Launch Jupyter

```bash
# Start Jupyter Notebook
jupyter notebook

# Or use JupyterLab
jupyter lab
```

Then navigate to the `notebooks/` directory and open any `.ipynb` file.

## Notebook Contents Overview

### 01_data_exploration.ipynb (90 cells)
- Data loading and initial inspection
- Missing value analysis
- RFM feature engineering from transactions
- Weekly aggregation over 105 weeks
- Distribution visualizations
- Customer segmentation hypothesis

### 02_clustering_experiments.ipynb (170 cells)
- **K-Means Clustering**
  - Elbow method (k=1 to 15)
  - Optimal k=4 selection
  - Silhouette Score: 0.677
  - Calinski-Harabasz: 101,980

- **DBSCAN Clustering**
  - Grid search over 30+ parameter combinations
  - eps and min_samples tuning
  - Silhouette Score: 0.459
  - High noise point ratio

- **Agglomerative Clustering**
  - Ward linkage method
  - Dendrogram visualization
  - Silhouette Score: 0.37-0.42

### 03_temporal_stability_simulation.ipynb (32 cells)
- Train K-Means on week 52 baseline
- Apply to weeks 53-104
- Track Adjusted Rand Index (ARI) over time
- Identify optimal retraining frequency
- Visualize model drift

## Reproducibility

All notebooks can be reproduced using the modular code in `src/`:

```python
# Instead of notebook-only code, use:
from src.data.data_loader import OlistDataLoader
from src.features.rfm_engineering import RFMCalculator
from src.models.clustering import CustomerSegmenter
from src.evaluation.metrics import evaluate_clustering
from src.visualization.plots import plot_clusters_3d_interactive

# This ensures consistency between notebooks and production code
```

## Notes

- Notebooks contain the original research and experimentation
- Production code in `src/` extracts the best practices from notebooks
- Notebooks are kept for transparency and reproducibility
- Cell outputs are preserved to show results without rerunning
