# 🛍️ Customer Segmentation for E-Commerce | Segmentation Client E-Commerce

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[🇬🇧 English](#english) | [🇫🇷 Français](#français)

---

<a name="english"></a>
## 🇬🇧 English Version

### 📋 Table of Contents
- [Overview](#overview)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Interactive Dashboard](#interactive-dashboard)
- [Technologies](#technologies)
- [Author](#author)

---

### 🎯 Overview

This project implements an **RFM-based customer segmentation system** for Olist, a Brazilian e-commerce platform. Using unsupervised machine learning (K-Means clustering), the solution identifies distinct customer segments to enable targeted marketing strategies and improve customer retention.

**Key Features:**
- 📊 RFM (Recency, Frequency, Monetary) feature engineering
- 🤖 Multiple clustering algorithms (K-Means, DBSCAN, Agglomerative)
- 📈 Comprehensive model evaluation (Silhouette Score, Calinski-Harabasz, Davies-Bouldin)
- 🎨 Interactive Streamlit dashboard for segment exploration
- ⏱️ Temporal stability analysis for model maintenance planning
- 🔄 Production-ready modular code architecture

---

### 💼 Business Problem

Olist needed to **differentiate customers** based on purchasing behavior and satisfaction to optimize marketing campaigns. With only ~3% of customers making repeat purchases, the challenge was to identify high-value segments and customers at risk of churn.

**Objectives:**
1. Segment customers into actionable groups
2. Enable personalized marketing strategies
3. Improve customer lifetime value (CLV)
4. Reduce churn rate through targeted interventions

---

### 📦 Dataset

**Source:** [Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle)

**Datasets Used:**
- **Orders** (99,442 orders)
- **Order Items** (112,651 items)
- **Customers** (99,442 unique customers)
- **Order Payments** (103,887 payment transactions)
- **Order Reviews** (104,720 reviews with satisfaction scores)
- **Products** (32,952 products)
- **Sellers** (3,096 sellers)
- **Geolocation** (1M+ Brazilian ZIP codes)

**Period:** September 2016 - August 2018

---

### 🔬 Methodology

#### 1️⃣ **RFM Feature Engineering**
- **Recency:** Days since last purchase
- **Frequency:** Number of orders placed
- **Monetary:** Total amount spent

#### 2️⃣ **Clustering Algorithms Tested**
| Algorithm | Silhouette Score | Clusters | Notes |
|-----------|------------------|----------|-------|
| **K-Means** | **0.677** ✅ | 4 | Best performance, clear separation |
| DBSCAN | 0.459 | Variable | Many noise points |
| Agglomerative | 0.37-0.42 | 4 | Poorest separation |

#### 3️⃣ **Model Evaluation**
- Elbow method for optimal K selection
- Silhouette analysis for cluster quality
- Calinski-Harabasz score for separation validation
- Temporal stability analysis (Adjusted Rand Index over 52 weeks)

---

### 🏆 Key Results

**4 Customer Segments Identified:**

| Segment | Size | Characteristics | Avg Recency | Avg Frequency | Avg Spend |
|---------|------|-----------------|-------------|---------------|-----------|
| **0: Recent Browsers** | 54% | Single-purchase, recent customers | ~132 days | 1.0 | $103 |
| **1: Loyal Repeat Buyers** | 3% | Multiple purchases, best retention | ~225 days | 2.1 | $211 |
| **2: Dormant/Inactive** | 40% | Old customers, churn risk | ~393 days | 1.0 | $103 |
| **3: VIP/High-Value** | 3% | Highest spenders | ~246 days | 1.0 | $1,017 |

**Model Performance:**
- **Silhouette Score:** 0.677 (excellent cluster separation)
- **Calinski-Harabasz Score:** 101,980 (strong inter-cluster variance)
- **Recommended Retraining Frequency:** Every 12-16 weeks (based on ARI stability)

---

### 📁 Project Structure

```
Segmenter_Clients_Ecom/
├── data/
│   ├── raw/                        # Olist CSV datasets (gitignored)
│   └── processed/                  # Processed RFM data
├── src/
│   ├── data/
│   │   └── data_loader.py          # Data loading utilities
│   ├── features/
│   │   └── rfm_engineering.py      # RFM calculation
│   ├── models/
│   │   └── clustering.py           # Clustering algorithms
│   ├── evaluation/
│   │   └── metrics.py              # Model evaluation
│   └── visualization/
│       └── plots.py                # Plotting functions
├── notebooks/                       # Jupyter notebooks (EDA, experiments)
├── outputs/
│   ├── models/                     # Saved trained models
│   └── figures/                    # Generated visualizations
├── docs/                           # Additional documentation
├── app.py                          # Streamlit dashboard
├── train.py                        # Training script
├── predict.py                      # Prediction script
├── evaluate.py                     # Evaluation script
├── config.yaml                     # Configuration file
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

### 🚀 Installation

#### Prerequisites
- Python 3.9 or higher
- pip or conda

#### Step 1: Clone the repository
```bash
git clone https://github.com/ThomasMeb/Segmenter_Clients_Ecom.git
cd Segmenter_Clients_Ecom
```

#### Step 2: Create virtual environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n customer-segmentation python=3.9
conda activate customer-segmentation
```

#### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

#### Step 4: Download Olist dataset
Download the [Olist dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) from Kaggle and place CSV files in `data/raw/`.

---

### 💻 Usage

#### 1️⃣ **Train the Model**
```bash
# Train with default configuration
python train.py

# Find optimal K with elbow method
python train.py --find-optimal-k

# Use custom config
python train.py --config my_config.yaml
```

#### 2️⃣ **Make Predictions**
```bash
# Predict clusters for new customers
python predict.py --input data/processed/new_customers.csv --output data/processed/predictions.csv
```

#### 3️⃣ **Evaluate Model**
```bash
# Generate evaluation report
python evaluate.py
```

#### 4️⃣ **Launch Dashboard**
```bash
streamlit run app.py
```

---

### 🎨 Interactive Dashboard

Launch the Streamlit dashboard to explore customer segments interactively:

**Features:**
- 📊 Cluster overview and distribution
- 🔍 3D RFM visualization
- 👥 Customer search and top customers
- 📈 RFM distribution analysis
- 📥 Export segment data to CSV

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`

---

### 🛠️ Technologies

**Core:**
- Python 3.9+
- pandas, NumPy
- scikit-learn (K-Means, DBSCAN, Agglomerative Clustering)
- scipy (hierarchical clustering)

**Visualization:**
- matplotlib, seaborn
- Plotly (interactive 3D plots)
- Streamlit (dashboard)
- Yellowbrick (ML visualization)

**Utilities:**
- PyYAML (configuration)
- joblib (model persistence)
- Jupyter (notebooks)

---

### 👤 Author

**Thomas Mebarki**
- LinkedIn: [thomas-mebarki](https://www.linkedin.com/in/thomas-mebarki)
- Email: thomas.mebarki@example.com

---

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<a name="français"></a>
## 🇫🇷 Version Française

### 📋 Table des matières
- [Aperçu](#aperçu-fr)
- [Problématique métier](#problématique-métier)
- [Données](#données)
- [Méthodologie](#méthodologie-fr)
- [Résultats clés](#résultats-clés)
- [Structure du projet](#structure-du-projet-fr)
- [Installation](#installation-fr)
- [Utilisation](#utilisation-fr)
- [Dashboard interactif](#dashboard-interactif)
- [Technologies](#technologies-fr)
- [Auteur](#auteur-fr)

---

<a name="aperçu-fr"></a>
### 🎯 Aperçu

Ce projet implémente un **système de segmentation client basé sur l'analyse RFM** pour Olist, une plateforme de e-commerce brésilienne. Utilisant l'apprentissage automatique non supervisé (clustering K-Means), la solution identifie des segments de clients distincts pour permettre des stratégies marketing ciblées et améliorer la rétention client.

**Fonctionnalités principales :**
- 📊 Ingénierie de features RFM (Récence, Fréquence, Montant)
- 🤖 Plusieurs algorithmes de clustering (K-Means, DBSCAN, Agglomératif)
- 📈 Évaluation complète du modèle (Silhouette, Calinski-Harabasz, Davies-Bouldin)
- 🎨 Dashboard Streamlit interactif pour l'exploration des segments
- ⏱️ Analyse de stabilité temporelle pour planifier la maintenance du modèle
- 🔄 Architecture de code modulaire prête pour la production

---

### 💼 Problématique métier

Olist avait besoin de **différencier ses clients** selon leur comportement d'achat et leur satisfaction pour optimiser les campagnes marketing. Avec seulement ~3% des clients effectuant des achats répétés, le défi était d'identifier les segments à forte valeur et les clients à risque de churn.

**Objectifs :**
1. Segmenter les clients en groupes actionnables
2. Permettre des stratégies marketing personnalisées
3. Améliorer la valeur vie client (CLV)
4. Réduire le taux de churn via des interventions ciblées

---

### 📦 Données

**Source :** [Dataset E-Commerce Brésilien Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle)

**Datasets utilisés :**
- **Commandes** (99 442 commandes)
- **Articles commandés** (112 651 articles)
- **Clients** (99 442 clients uniques)
- **Paiements** (103 887 transactions)
- **Avis clients** (104 720 avis avec scores de satisfaction)
- **Produits** (32 952 produits)
- **Vendeurs** (3 096 vendeurs)
- **Géolocalisation** (1M+ codes postaux brésiliens)

**Période :** Septembre 2016 - Août 2018

---

<a name="méthodologie-fr"></a>
### 🔬 Méthodologie

#### 1️⃣ **Ingénierie de features RFM**
- **Récence (Recency) :** Jours depuis le dernier achat
- **Fréquence (Frequency) :** Nombre de commandes passées
- **Montant (Monetary) :** Montant total dépensé

#### 2️⃣ **Algorithmes de clustering testés**
| Algorithme | Score Silhouette | Clusters | Notes |
|-----------|------------------|----------|-------|
| **K-Means** | **0.677** ✅ | 4 | Meilleure performance, séparation claire |
| DBSCAN | 0.459 | Variable | Nombreux points de bruit |
| Agglomératif | 0.37-0.42 | 4 | Séparation la plus faible |

#### 3️⃣ **Évaluation du modèle**
- Méthode du coude pour sélection du K optimal
- Analyse de silhouette pour qualité des clusters
- Score Calinski-Harabasz pour validation de la séparation
- Analyse de stabilité temporelle (Adjusted Rand Index sur 52 semaines)

---

### 🏆 Résultats clés

**4 segments clients identifiés :**

| Segment | Taille | Caractéristiques | Récence moy. | Fréquence moy. | Dépense moy. |
|---------|--------|------------------|--------------|----------------|--------------|
| **0 : Browsers récents** | 54% | Achat unique, clients récents | ~132 jours | 1.0 | 103$ |
| **1 : Fidèles récurrents** | 3% | Achats multiples, meilleure rétention | ~225 jours | 2.1 | 211$ |
| **2 : Dormants/Inactifs** | 40% | Anciens clients, risque de churn | ~393 jours | 1.0 | 103$ |
| **3 : VIP/Haute valeur** | 3% | Dépenses les plus élevées | ~246 jours | 1.0 | 1 017$ |

**Performance du modèle :**
- **Score Silhouette :** 0.677 (excellente séparation des clusters)
- **Score Calinski-Harabasz :** 101 980 (forte variance inter-clusters)
- **Fréquence de ré-entraînement recommandée :** Toutes les 12-16 semaines (basé sur stabilité ARI)

---

<a name="structure-du-projet-fr"></a>
### 📁 Structure du projet

Voir la section [Project Structure](#project-structure) ci-dessus.

---

<a name="installation-fr"></a>
### 🚀 Installation

Voir la section [Installation](#installation) ci-dessus.

---

<a name="utilisation-fr"></a>
### 💻 Utilisation

#### 1️⃣ **Entraîner le modèle**
```bash
# Entraînement avec configuration par défaut
python train.py

# Trouver le K optimal avec la méthode du coude
python train.py --find-optimal-k

# Utiliser une configuration personnalisée
python train.py --config ma_config.yaml
```

#### 2️⃣ **Faire des prédictions**
```bash
# Prédire les clusters pour de nouveaux clients
python predict.py --input data/processed/nouveaux_clients.csv --output data/processed/predictions.csv
```

#### 3️⃣ **Évaluer le modèle**
```bash
# Générer un rapport d'évaluation
python evaluate.py
```

#### 4️⃣ **Lancer le dashboard**
```bash
streamlit run app.py
```

---

### 🎨 Dashboard interactif

Lancez le dashboard Streamlit pour explorer les segments de clients de manière interactive.

**Fonctionnalités :**
- 📊 Vue d'ensemble et distribution des clusters
- 🔍 Visualisation 3D RFM
- 👥 Recherche de clients et top clients
- 📈 Analyse de distribution RFM
- 📥 Export des données de segments en CSV

```bash
streamlit run app.py
```

Puis ouvrez votre navigateur à l'adresse `http://localhost:8501`

---

<a name="technologies-fr"></a>
### 🛠️ Technologies

Voir la section [Technologies](#technologies) ci-dessus.

---

<a name="auteur-fr"></a>
### 👤 Auteur

**Thomas Mebarki**
- LinkedIn: [thomas-mebarki](https://www.linkedin.com/in/thomas-mebarki)
- Email: thomas.mebarki@example.com

---

### 📄 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 🙏 Remerciements

- **Olist** pour la mise à disposition du dataset sur Kaggle
- **OpenClassrooms** pour l'encadrement du projet initial
- La communauté Python data science pour les outils open-source

---

**⭐ Si ce projet vous a été utile, n'hésitez pas à lui donner une étoile sur GitHub !**
