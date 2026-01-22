# Spécifications Techniques — Projet Olist Customer Segmentation

> **Document créé le :** 2026-01-21
> **Auteur :** Thomas Mebarki
> **Version :** 1.0
> **Statut :** Draft pour refactoring portfolio

---

## Table des Matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Contexte métier](#2-contexte-métier)
3. [Architecture des données](#3-architecture-des-données)
4. [Logique métier — Segmentation RFM](#4-logique-métier--segmentation-rfm)
5. [Modèles de Machine Learning](#5-modèles-de-machine-learning)
6. [Stack technique](#6-stack-technique)
7. [Structure cible du projet](#7-structure-cible-du-projet)
8. [Refactoring nécessaire](#8-refactoring-nécessaire)
9. [Critères de qualité](#9-critères-de-qualité)
10. [Roadmap](#10-roadmap)

---

## 1. Vue d'ensemble

### 1.1 Objectif du projet

Segmenter les clients d'**Olist**, une plateforme e-commerce brésilienne, afin de :
- Identifier les profils clients (VIP, fidèles, à risque, dormants)
- Permettre des actions marketing ciblées
- Proposer un contrat de maintenance pour la mise à jour du modèle

### 1.2 Objectif portfolio

Transformer ce projet académique en **projet portfolio professionnel** démontrant :
- Maîtrise du Machine Learning non supervisé
- Compétences en clean code et architecture
- Capacité à produire un livrable industrialisable

---

## 2. Contexte métier

### 2.1 Olist en bref

| Attribut | Valeur |
|----------|--------|
| **Secteur** | E-commerce / Marketplace |
| **Pays** | Brésil |
| **Modèle** | B2B2C (vendeurs → Olist → clients) |
| **Volume** | ~100,000 commandes (dataset) |
| **Période** | Sept 2016 — Sept 2018 |

### 2.2 Problématique métier

> *"Comment segmenter efficacement nos clients pour optimiser nos campagnes marketing ?"*

**Contraintes :**
- 97% des clients n'ont qu'une seule commande
- Données limitées (pas de données démographiques)
- Besoin d'une segmentation simple et actionnable

### 2.3 Parties prenantes

| Rôle | Besoin |
|------|--------|
| **Équipe Marketing** | Segments clairs pour campagnes ciblées |
| **Direction** | ROI des actions marketing |
| **Data Team** | Modèle maintenable et reproductible |

---

## 3. Architecture des données

### 3.1 Sources de données

```
data/
├── data.csv           # Transactions brutes (100k lignes)
├── data_2.csv         # Transactions + review_score (95k lignes)
└── data_RFM.csv       # Features RFM pré-calculées (95k lignes)
```

### 3.2 Schéma des données brutes (`data.csv`)

| Colonne | Type | Description | Exemple |
|---------|------|-------------|---------|
| `customer_id` | string | ID unique de la transaction client | `9ef432eb6251...` |
| `customer_unique_id` | string | ID unique du client | `7c396fd4830f...` |
| `order_id` | string | ID de la commande | `e481f51cbdc5...` |
| `price` | float | Montant de la commande | 29.99 |
| `order_purchase_timestamp` | datetime | Date d'achat | `2017-10-02 10:56:33` |

### 3.3 Schéma RFM (`data_RFM.csv`)

| Colonne | Type | Description | Calcul |
|---------|------|-------------|--------|
| `customer_unique_id` | string | ID client (clé) | — |
| `frequency` | int | Nombre de commandes | `COUNT(order_id)` |
| `recency` | int | Jours depuis dernière commande | `MAX_DATE - MAX(order_date)` |
| `amount_spent` | float | Montant total dépensé | `SUM(price)` |

### 3.4 Statistiques clés

| Métrique | Valeur |
|----------|--------|
| **Clients uniques** | 95,420 |
| **Clients multi-commandes** | 2,913 (3%) |
| **Période couverte** | 728 jours |
| **Montant moyen** | 130.25 BRL |
| **Fréquence moyenne** | 1.03 commandes |

---

## 4. Logique métier — Segmentation RFM

### 4.1 Principe RFM

Le modèle **RFM** (Recency, Frequency, Monetary) est une technique de segmentation client basée sur le comportement d'achat :

```
┌─────────────────────────────────────────────────────────────┐
│                      SEGMENTATION RFM                        │
├─────────────┬─────────────┬─────────────────────────────────┤
│  RECENCY    │  FREQUENCY  │  MONETARY (Amount Spent)        │
│  (R)        │  (F)        │  (M)                            │
├─────────────┼─────────────┼─────────────────────────────────┤
│  Quand ?    │  Combien    │  Combien dépensé ?              │
│  Dernier    │  de fois ?  │                                 │
│  achat      │             │                                 │
└─────────────┴─────────────┴─────────────────────────────────┘
```

### 4.2 Calcul des features

```python
# Recency : jours depuis le dernier achat
recency = (date_reference - date_dernier_achat).days

# Frequency : nombre total de commandes
frequency = df.groupby('customer_unique_id')['order_id'].count()

# Monetary : montant total dépensé
amount_spent = df.groupby('customer_unique_id')['price'].sum()
```

### 4.3 Segments identifiés

| Cluster | Nom suggéré | Profil | Taille | Action marketing |
|---------|-------------|--------|--------|------------------|
| **0** | Récents | R↓ F↓ M↓ | 54% | Fidélisation |
| **1** | Fidèles | R~ F↑ M↑ | 3% | Programme VIP |
| **2** | Dormants | R↑ F↓ M↓ | 40% | Réactivation |
| **3** | VIP | R~ F↓ M↑↑ | 3% | Rétention premium |

---

## 5. Modèles de Machine Learning

### 5.1 Comparatif des modèles testés

| Modèle | Silhouette Score | Calinski-Harabasz | Verdict |
|--------|------------------|-------------------|---------|
| **KMeans (k=4)** | **0.677** | **101,980** | ✅ Retenu |
| DBSCAN | 0.459 | 1,128 | ❌ Trop de bruit |
| Agglomerative (k=4) | 0.418 | — | ❌ Moins performant |

### 5.2 Configuration du modèle retenu

```python
# Modèle final
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[['recency', 'frequency', 'amount_spent']])

# Clustering
model = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10,
    max_iter=300
)
clusters = model.fit_predict(X_scaled)
```

### 5.3 Validation du modèle

**Métriques utilisées :**
- **Silhouette Score** : Mesure la cohésion intra-cluster et séparation inter-cluster
- **Calinski-Harabasz Index** : Ratio variance inter/intra cluster
- **Elbow Method** : Détermination du nombre optimal de clusters

### 5.4 Maintenance du modèle

**Analyse de stabilité temporelle (ARI Score) :**

| Intervalle | ARI Score | Recommandation |
|------------|-----------|----------------|
| 1-10 semaines | > 0.7 | ✅ Stable |
| 10-20 semaines | 0.4-0.7 | ⚠️ Surveillance |
| > 20 semaines | < 0.4 | 🔄 Réentraînement |

**Fréquence de mise à jour recommandée : 3 mois**

---

## 6. Stack technique

### 6.1 Dépendances Python

```txt
# Core
pandas>=1.5.0
numpy>=1.23.0

# Machine Learning
scikit-learn>=1.2.0

# Visualization
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.11.0

# Clustering visualization
yellowbrick>=1.5

# Hierarchical clustering
scipy>=1.9.0

# Development
jupyter>=1.0.0
ipykernel>=6.0.0
```

### 6.2 Version Python

```
Python 3.10+
```

### 6.3 Outils de qualité (à ajouter)

```txt
# Linting & Formatting
black>=23.0.0
isort>=5.12.0
flake8>=6.0.0

# Type checking
mypy>=1.0.0

# Testing
pytest>=7.0.0
pytest-cov>=4.0.0
```

---

## 7. Structure cible du projet

### 7.1 Arborescence proposée

```
olist-customer-segmentation/
│
├── README.md                    # Documentation principale (portfolio)
├── LICENSE                      # MIT License
├── pyproject.toml              # Configuration projet & dépendances
├── requirements.txt            # Dépendances (pip)
│
├── data/
│   ├── raw/                    # Données brutes (gitignore)
│   │   ├── data.csv
│   │   └── data_2.csv
│   ├── processed/              # Données transformées
│   │   └── data_rfm.csv
│   └── README.md               # Description des données
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py           # Chargement des données
│   │   └── preprocessor.py     # Preprocessing & feature engineering
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── rfm_calculator.py   # Calcul des features RFM
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── clustering.py       # Modèles de clustering
│   │   └── evaluation.py       # Métriques d'évaluation
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plots.py            # Fonctions de visualisation
│   │
│   └── utils/
│       ├── __init__.py
│       └── config.py           # Configuration globale
│
├── notebooks/
│   ├── 01_exploration.ipynb    # EDA propre et commenté
│   ├── 02_modeling.ipynb       # Entraînement et comparaison
│   └── 03_results.ipynb        # Résultats et visualisations finales
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessor.py
│   ├── test_rfm.py
│   └── test_clustering.py
│
├── docs/
│   ├── SPECIFICATIONS_TECHNIQUES.md
│   ├── ARCHITECTURE.md
│   └── images/
│       └── segments_radar.png
│
└── .github/
    └── workflows/
        └── ci.yml              # GitHub Actions CI/CD
```

### 7.2 Modules principaux

| Module | Responsabilité |
|--------|----------------|
| `data.loader` | Chargement et validation des données |
| `data.preprocessor` | Nettoyage, transformation |
| `features.rfm_calculator` | Calcul des features RFM |
| `models.clustering` | Entraînement des modèles |
| `models.evaluation` | Métriques et validation |
| `visualization.plots` | Graphiques standardisés |

---

## 8. Refactoring nécessaire

### 8.1 Code à extraire en fonctions

```python
# Exemple : rfm_calculator.py

def calculate_rfm(
    df: pd.DataFrame,
    customer_col: str = 'customer_unique_id',
    date_col: str = 'order_purchase_timestamp',
    amount_col: str = 'price',
    reference_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Calculate RFM features for customer segmentation.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction data with customer, date, and amount columns.
    customer_col : str
        Name of the customer identifier column.
    date_col : str
        Name of the transaction date column.
    amount_col : str
        Name of the transaction amount column.
    reference_date : datetime, optional
        Reference date for recency calculation. Defaults to max date in data.

    Returns
    -------
    pd.DataFrame
        DataFrame with customer_id, recency, frequency, and monetary columns.

    Examples
    --------
    >>> rfm = calculate_rfm(transactions, reference_date=datetime(2018, 9, 1))
    >>> rfm.head()
    """
    pass
```

### 8.2 Patterns à appliquer

| Pattern | Application |
|---------|-------------|
| **Single Responsibility** | Une fonction = une tâche |
| **DRY** | Éliminer la duplication de code |
| **Type Hints** | Typage explicite des fonctions |
| **Docstrings** | Documentation Google/NumPy style |
| **Constants** | Configuration dans fichier dédié |

### 8.3 Problèmes à corriger

| Problème | Fichier | Action |
|----------|---------|--------|
| Cellules vides | `notebook_essais` | Supprimer |
| IndexError | `notebook_essais:cell-52` | Corriger la fonction |
| ValueError | `notebook_essais:cell-110` | Corriger les dimensions |
| Code dupliqué | Tous | Factoriser en fonctions |
| Warnings ignorés | Tous | Traiter les warnings |

---

## 9. Critères de qualité

### 9.1 Checklist Clean Code

- [ ] Noms de variables explicites (`customer_rfm` vs `data`)
- [ ] Fonctions < 20 lignes
- [ ] Une fonction = un seul niveau d'abstraction
- [ ] Type hints sur toutes les fonctions publiques
- [ ] Docstrings format NumPy/Google
- [ ] Pas de magic numbers (utiliser des constantes)
- [ ] Gestion des erreurs appropriée
- [ ] Pas de code commenté

### 9.2 Checklist Tests

- [ ] Coverage > 80% sur `src/`
- [ ] Tests unitaires pour chaque fonction
- [ ] Tests d'intégration pour le pipeline
- [ ] Tests de validation des données

### 9.3 Checklist Documentation

- [ ] README avec badges, screenshots, instructions
- [ ] Docstrings complètes
- [ ] Architecture documentée
- [ ] Guide de contribution

### 9.4 Checklist CI/CD

- [ ] GitHub Actions : lint + tests
- [ ] Pre-commit hooks
- [ ] Badge de coverage

---

## 10. Roadmap

### Phase 1 : Audit ✅
- [x] Analyse des notebooks existants
- [x] Identification des dépendances
- [x] Documentation de la logique métier
- [x] Création des spécifications techniques

### Phase 2 : Architecture
- [ ] Création de la structure de dossiers
- [ ] Setup du projet (pyproject.toml)
- [ ] Configuration des outils de qualité

### Phase 3 : Refactoring
- [ ] Extraction du code en modules Python
- [ ] Ajout des type hints et docstrings
- [ ] Création des notebooks propres

### Phase 4 : Tests
- [ ] Tests unitaires
- [ ] Tests d'intégration
- [ ] Setup CI/CD

### Phase 5 : Documentation
- [ ] README portfolio-ready
- [ ] Documentation technique
- [ ] Visualisations finales

### Phase 6 : Polish
- [ ] Optimisation des visualisations
- [ ] Notebook de démonstration
- [ ] (Optionnel) Dashboard Streamlit

---

## Annexes

### A. Glossaire

| Terme | Définition |
|-------|------------|
| **RFM** | Recency, Frequency, Monetary — méthode de segmentation |
| **Silhouette Score** | Métrique de qualité de clustering [-1, 1] |
| **ARI** | Adjusted Rand Index — mesure de similarité entre clusterings |
| **Elbow Method** | Technique pour déterminer le nombre optimal de clusters |

### B. Références

- [Olist Dataset (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
- [Scikit-learn Clustering Documentation](https://scikit-learn.org/stable/modules/clustering.html)
- [RFM Analysis Wikipedia](https://en.wikipedia.org/wiki/RFM_(market_research))

---

*Document généré par Mary — Business Analyst Agent (BMad Method)*
