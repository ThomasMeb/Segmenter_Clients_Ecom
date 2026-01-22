# Architecture du Projet — Olist Customer Segmentation

> **Version :** 1.0
> **Architecte :** Winston (BMad Architect Agent)
> **Date :** 2026-01-21

---

## 1. Vue d'ensemble

### 1.1 Diagramme de haut niveau

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     OLIST CUSTOMER SEGMENTATION                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │     DATA     │───▶│   PIPELINE   │───▶│    MODEL     │               │
│  │    (LFS)     │    │  (src/*)     │    │  (KMeans)    │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
│         │                   │                   │                        │
│         ▼                   ▼                   ▼                        │
│  ┌──────────────────────────────────────────────────────┐               │
│  │                   STREAMLIT DASHBOARD                 │               │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  │               │
│  │  │ Overview│  │Segments │  │ Explore │  │  About  │  │               │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  │               │
│  └──────────────────────────────────────────────────────┘               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Choix architecturaux

| Décision | Choix | Justification |
|----------|-------|---------------|
| Structure | Standard | Équilibre lisibilité/professionnalisme |
| Données | Git LFS | Versioning propre, repo clonable |
| Interface | Streamlit | Rapide à développer, interactif, déployable |
| ML Framework | Scikit-learn | Standard industrie, suffisant pour clustering |
| Config | pyproject.toml | Standard moderne Python |

---

## 2. Structure du projet

```
olist-customer-segmentation/
│
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions : lint + tests
│
├── .gitattributes                 # Configuration Git LFS
├── .gitignore                     # Fichiers ignorés
├── .pre-commit-config.yaml        # Hooks pre-commit
│
├── LICENSE                        # MIT License
├── README.md                      # Documentation principale
├── pyproject.toml                 # Config projet + dépendances
├── requirements.txt               # Dépendances (compatibilité pip)
│
├── data/
│   ├── .gitkeep
│   ├── README.md                  # Description des données
│   ├── raw/                       # Données brutes (Git LFS)
│   │   ├── olist_customers.csv
│   │   ├── olist_orders.csv
│   │   └── olist_order_payments.csv
│   └── processed/                 # Données transformées
│       └── customers_rfm.parquet
│
├── models/                        # Modèles sauvegardés
│   ├── .gitkeep
│   ├── kmeans_model.pkl
│   └── scaler.pkl
│
├── src/
│   ├── __init__.py
│   │
│   ├── config.py                  # Configuration globale
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py              # Chargement des données
│   │   └── preprocessor.py        # Nettoyage & transformation
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── rfm.py                 # Calcul features RFM
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── clustering.py          # Modèles de clustering
│   │   └── evaluation.py          # Métriques & validation
│   │
│   └── visualization/
│       ├── __init__.py
│       └── plots.py               # Fonctions de visualisation
│
├── notebooks/
│   ├── 01_data_exploration.ipynb  # EDA propre
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb          # Entraînement & comparaison
│   └── 04_results_analysis.ipynb  # Analyse des résultats
│
├── app/                           # Dashboard Streamlit
│   ├── __init__.py
│   ├── app.py                     # Point d'entrée
│   ├── pages/
│   │   ├── 1_Overview.py          # Vue d'ensemble
│   │   ├── 2_Segments.py          # Détail des segments
│   │   ├── 3_Explorer.py          # Exploration interactive
│   │   └── 4_About.py             # À propos du projet
│   ├── components/
│   │   ├── __init__.py
│   │   ├── charts.py              # Composants graphiques
│   │   └── sidebar.py             # Sidebar commune
│   └── assets/
│       ├── style.css              # Styles custom
│       └── logo.png               # Logo projet
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Fixtures pytest
│   ├── test_data/
│   │   └── sample_data.csv        # Données de test
│   ├── test_loader.py
│   ├── test_rfm.py
│   └── test_clustering.py
│
└── docs/
    ├── ARCHITECTURE.md            # Ce document
    ├── SPECIFICATIONS_TECHNIQUES.md
    └── images/
        ├── segments_radar.png
        └── dashboard_preview.png
```

---

## 3. Composants détaillés

### 3.1 Module `src/config.py`

```python
"""Configuration globale du projet."""
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"

# Model parameters
N_CLUSTERS = 4
RANDOM_STATE = 42

# Feature names
RFM_FEATURES = ["recency", "frequency", "monetary"]

# Segment names
SEGMENT_NAMES = {
    0: "Récents",
    1: "Fidèles",
    2: "Dormants",
    3: "VIP"
}
```

### 3.2 Module `src/data/loader.py`

```python
"""Chargement et validation des données."""
from pathlib import Path
from typing import Optional
import pandas as pd

def load_transactions(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Charge les données de transactions.

    Parameters
    ----------
    filepath : Path, optional
        Chemin vers le fichier. Par défaut: data/raw/olist_orders.csv

    Returns
    -------
    pd.DataFrame
        DataFrame des transactions avec colonnes validées.

    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas.
    ValueError
        Si les colonnes requises sont manquantes.
    """
    pass
```

### 3.3 Module `src/features/rfm.py`

```python
"""Calcul des features RFM pour la segmentation client."""
from datetime import datetime
from typing import Optional
import pandas as pd

class RFMCalculator:
    """
    Calculateur de features RFM (Recency, Frequency, Monetary).

    Attributes
    ----------
    reference_date : datetime
        Date de référence pour le calcul de la récence.

    Examples
    --------
    >>> calculator = RFMCalculator(reference_date=datetime(2018, 9, 1))
    >>> rfm_df = calculator.fit_transform(transactions_df)
    """

    def __init__(self, reference_date: Optional[datetime] = None):
        self.reference_date = reference_date

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule les features RFM à partir des transactions."""
        pass
```

### 3.4 Module `src/models/clustering.py`

```python
"""Modèles de clustering pour la segmentation client."""
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class CustomerSegmenter:
    """
    Pipeline de segmentation client basé sur KMeans.

    Attributes
    ----------
    n_clusters : int
        Nombre de clusters (segments).
    scaler : StandardScaler
        Scaler pour normalisation des features.
    model : KMeans
        Modèle KMeans entraîné.

    Examples
    --------
    >>> segmenter = CustomerSegmenter(n_clusters=4)
    >>> segmenter.fit(rfm_df)
    >>> labels = segmenter.predict(new_customers)
    >>> segmenter.save("models/")
    """

    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, X: pd.DataFrame) -> "CustomerSegmenter":
        """Entraîne le modèle de segmentation."""
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Prédit les segments pour de nouveaux clients."""
        pass

    def save(self, path: str) -> None:
        """Sauvegarde le modèle et le scaler."""
        pass

    @classmethod
    def load(cls, path: str) -> "CustomerSegmenter":
        """Charge un modèle sauvegardé."""
        pass
```

---

## 4. Dashboard Streamlit

### 4.1 Architecture des pages

```
app/
├── app.py                    # Point d'entrée principal
│   └── Configuration Streamlit
│   └── Chargement des données en cache
│   └── Navigation multi-pages
│
└── pages/
    ├── 1_Overview.py         # Page d'accueil
    │   └── KPIs principaux
    │   └── Distribution des segments
    │   └── Radar chart comparatif
    │
    ├── 2_Segments.py         # Détail par segment
    │   └── Sélecteur de segment
    │   └── Profil détaillé
    │   └── Clients représentatifs
    │   └── Recommandations marketing
    │
    ├── 3_Explorer.py         # Exploration interactive
    │   └── Filtres dynamiques
    │   └── Scatter plot interactif
    │   └── Export de données
    │
    └── 4_About.py            # À propos
        └── Description du projet
        └── Méthodologie
        └── Liens GitHub/LinkedIn
```

### 4.2 Wireframe Dashboard

```
┌─────────────────────────────────────────────────────────────────────┐
│  🛒 Olist Customer Segmentation                    [Theme] [GitHub] │
├──────────────┬──────────────────────────────────────────────────────┤
│              │                                                       │
│  NAVIGATION  │   📊 OVERVIEW                                        │
│              │                                                       │
│  ○ Overview  │   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  ○ Segments  │   │ 95,420  │ │  3.0%   │ │ 130 BRL │ │  0.677  │   │
│  ○ Explorer  │   │ Clients │ │ Fidèles │ │ Panier  │ │Silhouette│  │
│  ○ About     │   └─────────┘ └─────────┘ └─────────┘ └─────────┘   │
│              │                                                       │
│  ──────────  │   ┌──────────────────┐  ┌──────────────────────┐    │
│              │   │                  │  │                      │    │
│  FILTERS     │   │  PIE CHART       │  │   RADAR CHART        │    │
│              │   │  Segments        │  │   Profils            │    │
│  Date range  │   │                  │  │                      │    │
│  [────────]  │   └──────────────────┘  └──────────────────────┘    │
│              │                                                       │
│  Segment     │   ┌─────────────────────────────────────────────┐   │
│  [All     ▼] │   │                                             │   │
│              │   │         SEGMENT DETAILS TABLE               │   │
│              │   │                                             │   │
│              │   └─────────────────────────────────────────────┘   │
│              │                                                       │
└──────────────┴──────────────────────────────────────────────────────┘
```

---

## 5. Flux de données

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA FLOW                                    │
└─────────────────────────────────────────────────────────────────────┘

     RAW DATA                PROCESSING              MODEL
    ┌────────┐              ┌────────┐            ┌────────┐
    │ orders │──┐           │        │            │        │
    └────────┘  │           │  RFM   │   fit()    │ KMeans │
    ┌────────┐  ├──────────▶│ Calc.  │───────────▶│ k=4    │
    │customer│──┤           │        │            │        │
    └────────┘  │           └────────┘            └────────┘
    ┌────────┐  │               │                     │
    │payments│──┘               │                     │
    └────────┘                  ▼                     ▼
                          ┌────────┐            ┌────────┐
                          │  RFM   │  predict() │ labels │
                          │ .parquet│◀──────────│ .pkl   │
                          └────────┘            └────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  STREAMLIT DASHBOARD │
                    │   - Visualisations   │
                    │   - Interactions     │
                    │   - Export           │
                    └─────────────────────────┘
```

---

## 6. Configuration Git LFS

### 6.1 Fichiers trackés par LFS

```gitattributes
# Données
*.csv filter=lfs diff=lfs merge=lfs -text
*.parquet filter=lfs diff=lfs merge=lfs -text

# Modèles
*.pkl filter=lfs diff=lfs merge=lfs -text
*.joblib filter=lfs diff=lfs merge=lfs -text

# Images volumineuses
*.png filter=lfs diff=lfs merge=lfs -text
*.jpg filter=lfs diff=lfs merge=lfs -text
```

### 6.2 Setup Git LFS

```bash
# Installation
git lfs install

# Tracking des fichiers
git lfs track "*.csv"
git lfs track "*.parquet"
git lfs track "*.pkl"

# Vérification
git lfs ls-files
```

---

## 7. CI/CD Pipeline

### 7.1 GitHub Actions

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          lfs: true

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"

      - name: Lint with ruff
        run: ruff check src/ tests/

      - name: Format check with black
        run: black --check src/ tests/

      - name: Run tests
        run: pytest tests/ -v --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

---

## 8. Déploiement Streamlit

### 8.1 Options de déploiement

| Plateforme | Coût | Difficulté | URL |
|------------|------|------------|-----|
| **Streamlit Cloud** | Gratuit | ⭐ | `*.streamlit.app` |
| Hugging Face Spaces | Gratuit | ⭐⭐ | `*.hf.space` |
| Railway | Freemium | ⭐⭐ | Custom |
| Render | Freemium | ⭐⭐ | Custom |

### 8.2 Configuration Streamlit Cloud

```toml
# .streamlit/config.toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
headless = true
port = 8501
```

---

## 9. Décisions techniques

| # | Décision | Alternatives considérées | Raison du choix |
|---|----------|--------------------------|-----------------|
| 1 | Scikit-learn pour clustering | PyCaret, Rapids | Standard, léger, suffisant |
| 2 | Parquet pour données traitées | CSV, Feather | Compression, typage |
| 3 | Streamlit pour dashboard | Dash, Panel, Gradio | Simplicité, communauté |
| 4 | Ruff pour linting | Flake8, Pylint | Rapide, moderne |
| 5 | Git LFS pour données | DVC, externe | Simple, intégré GitHub |

---

## 10. Prochaines étapes

1. [ ] Créer la structure de dossiers
2. [ ] Configurer pyproject.toml
3. [ ] Initialiser Git LFS
4. [ ] Migrer le code existant vers les modules
5. [ ] Créer les notebooks propres
6. [ ] Développer le dashboard Streamlit
7. [ ] Écrire les tests
8. [ ] Configurer CI/CD
9. [ ] Déployer sur Streamlit Cloud

---

*Document créé par Winston — Architect Agent (BMad Method)*
