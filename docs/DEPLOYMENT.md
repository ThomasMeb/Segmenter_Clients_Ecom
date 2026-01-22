# Guide de Déploiement - Streamlit Cloud

Ce guide explique comment déployer le dashboard sur Streamlit Cloud.

## Prérequis

1. Un compte GitHub avec le projet pushé
2. Un compte Streamlit Cloud (gratuit) : https://share.streamlit.io

## Étapes de déploiement

### 1. Préparer le repository GitHub

```bash
# S'assurer que tous les fichiers sont commités
cd /home/thomas/Code/Projet_4
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
git push origin main
```

### 2. Configurer Streamlit Cloud

1. Aller sur https://share.streamlit.io
2. Se connecter avec GitHub
3. Cliquer sur "New app"
4. Sélectionner :
   - **Repository** : votre-username/olist-customer-segmentation
   - **Branch** : main
   - **Main file path** : app/app.py
5. Cliquer sur "Deploy!"

### 3. Configuration avancée (optionnel)

Si vous avez besoin de variables d'environnement ou de secrets :

1. Dans Streamlit Cloud, aller dans "Settings" > "Secrets"
2. Ajouter vos secrets au format TOML :

```toml
# .streamlit/secrets.toml (exemple)
[database]
host = "localhost"
password = "xxx"
```

## Fichiers importants pour le déploiement

| Fichier | Description |
|---------|-------------|
| `requirements.txt` | Dépendances Python |
| `app/app.py` | Point d'entrée de l'application |
| `.streamlit/config.toml` | Configuration Streamlit |

## Mode Démonstration

Le dashboard inclut un mode démonstration automatique :
- Si les données RFM réelles ne sont pas disponibles, des données de démo sont générées
- Cela permet au dashboard de fonctionner même sans les fichiers de données volumineux

Pour utiliser les vraies données sur Streamlit Cloud :

### Option A : Inclure les données prétraitées (recommandé)

1. Exécuter localement :
```bash
python scripts/prepare_dashboard_data.py
```

2. Commiter le fichier généré :
```bash
git add data/processed/customers_rfm.parquet
git commit -m "Add processed RFM data"
git push
```

### Option B : Générer les données au démarrage

Modifier `app/app.py` pour générer les données au premier lancement.
(Non recommandé car lent et consomme des ressources)

## Vérification du déploiement

Après le déploiement, vérifier que :

1. ✅ La page d'accueil s'affiche correctement
2. ✅ Les 4 pages de navigation fonctionnent
3. ✅ Les graphiques Plotly sont interactifs
4. ✅ Les filtres de l'Explorer fonctionnent
5. ✅ L'export CSV fonctionne

## Troubleshooting

### Erreur "No module named 'src'"

Ajouter ces lignes en haut de chaque fichier Python :
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Erreur de mémoire

Streamlit Cloud gratuit a des limites de ressources. Solutions :
- Réduire la taille des données
- Utiliser `@st.cache_data` pour le caching
- Échantillonner les données pour les visualisations

### Fichiers Git LFS non disponibles

Les fichiers Git LFS ne sont pas automatiquement téléchargés sur Streamlit Cloud.
Utilisez des fichiers plus petits ou le mode démonstration.

## URL du dashboard

Une fois déployé, votre dashboard sera accessible à :
```
https://votre-app-name.streamlit.app
```

## Mise à jour

Pour mettre à jour le dashboard déployé :
```bash
git add .
git commit -m "Update dashboard"
git push
```

Streamlit Cloud détectera automatiquement les changements et redéploiera.

---

## Ressources

- [Documentation Streamlit Cloud](https://docs.streamlit.io/streamlit-community-cloud)
- [Gestion des secrets](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management)
- [Limites de ressources](https://docs.streamlit.io/streamlit-community-cloud/manage-your-app#app-resources-and-limits)
