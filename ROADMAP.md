# Roadmap - Olist Customer Segmentation

> Plan d'amélioration du projet pour atteindre un niveau production-ready

---

## Vue d'ensemble

```
ÉTAT ACTUEL                           OBJECTIF
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Couverture tests    ~40%      ──────►    80%+
Code quality        7/10      ──────►    9/10
CI/CD               CI only   ──────►    CI + CD complet
CLI                 Absente   ──────►    Fonctionnelle
Documentation       4/5       ──────►    5/5
Monitoring          Aucun     ──────►    Drift detection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Phase 1 : Fondations & Qualité de Code

**Objectif** : Assainir la base de code et mettre en place les garde-fous

### 1.1 Refactoring des imports
- [ ] Supprimer tous les `sys.path.insert()` dans `app/pages/*.py`
- [ ] Configurer le package correctement avec `pip install -e .`
- [ ] Utiliser des imports absolus : `from src.config import ...`

**Fichiers concernés** :
- `app/pages/1_Overview.py`
- `app/pages/2_Segments.py`
- `app/pages/3_Explorer.py`
- `app/app.py`

### 1.2 Pre-commit hooks
- [ ] Créer `.pre-commit-config.yaml`
- [ ] Configurer : ruff, black, mypy, pytest (quick)
- [ ] Documenter dans README.md

```yaml
# .pre-commit-config.yaml (à créer)
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.6
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.0
    hooks:
      - id: mypy
        additional_dependencies: [pandas-stubs, types-requests]
```

### 1.3 Tests manquants - Partie 1
- [ ] Tests pour `src/data/preprocessor.py` (~15 tests)
- [ ] Tests pour `src/models/evaluation.py` (~10 tests)

**Critère de succès** : Couverture > 60%

| Livrable | Effort estimé |
|----------|---------------|
| Refactoring imports | 2h |
| Pre-commit hooks | 1h |
| Tests preprocessor | 3h |
| Tests evaluation | 2h |
| **Total Phase 1** | **8h** |

---

## Phase 2 : Tests & Robustesse

**Objectif** : Atteindre 80% de couverture et améliorer la fiabilité

### 2.1 Tests de visualisation
- [ ] Tests pour `src/visualization/plots.py`
- [ ] Utiliser `pytest-mpl` ou mock matplotlib
- [ ] Tester la génération sans affichage (backend `Agg`)

### 2.2 Tests du dashboard
- [ ] Tests pour `app/utils.py`
- [ ] Tests d'intégration avec `streamlit.testing` (Streamlit 1.28+)
- [ ] Tests des pages principales (smoke tests)

### 2.3 Error handling
- [ ] Ajouter try/except robustes dans le data loading
- [ ] Validation des inputs utilisateur dans le dashboard
- [ ] Messages d'erreur explicites et logging structuré

### 2.4 Tests d'intégration
- [ ] Test e2e : load → preprocess → train → predict → visualize
- [ ] Test de régression du modèle (performance minimale)

| Livrable | Effort estimé |
|----------|---------------|
| Tests visualisation | 3h |
| Tests dashboard | 4h |
| Error handling | 2h |
| Tests intégration | 3h |
| **Total Phase 2** | **12h** |

**Critère de succès** : Couverture > 80%, 0 crash sur données invalides

---

## Phase 3 : CLI & Automatisation

**Objectif** : Permettre l'utilisation en ligne de commande

### 3.1 Implémentation CLI
- [ ] Créer `src/cli.py` avec Click ou Typer
- [ ] Commandes : `train`, `predict`, `evaluate`, `serve`
- [ ] Options : `--input`, `--output`, `--n-clusters`, `--verbose`

```bash
# Usage cible
olist-segment train --input data/raw/data.csv --output models/
olist-segment predict --model models/ --input new_customers.csv
olist-segment serve --port 8501
```

### 3.2 Scripts d'automatisation
- [ ] `scripts/setup.sh` : Installation complète
- [ ] `scripts/train.sh` : Pipeline d'entraînement
- [ ] `scripts/download_data.sh` : Téléchargement Kaggle

### 3.3 Makefile
- [ ] Créer `Makefile` avec targets standards

```makefile
# Makefile (à créer)
.PHONY: install test lint train serve clean

install:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	ruff check src/ tests/ app/
	mypy src/

train:
	python -m src.cli train --input data/raw/data.csv

serve:
	streamlit run app/app.py

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache htmlcov/
```

| Livrable | Effort estimé |
|----------|---------------|
| CLI complète | 4h |
| Scripts automation | 2h |
| Makefile | 1h |
| **Total Phase 3** | **7h** |

---

## Phase 4 : CI/CD Complet

**Objectif** : Automatiser le déploiement

### 4.1 GitHub Actions - CD
- [ ] Déploiement auto sur Streamlit Cloud (main branch)
- [ ] Déploiement staging sur PR (preview)
- [ ] Notifications Slack/Discord (optionnel)

### 4.2 Versioning sémantique
- [ ] Configurer `python-semantic-release`
- [ ] Tags automatiques sur merge to main
- [ ] Génération automatique du CHANGELOG

### 4.3 Badges et status
- [ ] Badge couverture (Codecov)
- [ ] Badge build status
- [ ] Badge version

```yaml
# .github/workflows/cd.yml (à créer)
name: CD
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Deploy to Streamlit Cloud
        # Configuration Streamlit Cloud API
```

| Livrable | Effort estimé |
|----------|---------------|
| CD Pipeline | 3h |
| Semantic release | 2h |
| Badges & reporting | 1h |
| **Total Phase 4** | **6h** |

---

## Phase 5 : Documentation & Contribution ✅

**Objectif** : Faciliter les contributions et la maintenance

### 5.1 Documentation contributeur
- [x] Créer `CONTRIBUTING.md`
- [x] Créer `CODE_OF_CONDUCT.md`
- [x] Documenter le workflow Git (branches, PR, reviews)

### 5.2 Changelog
- [x] Créer `CHANGELOG.md`
- [x] Documenter toutes les versions passées
- [x] Configurer génération automatique

### 5.3 Documentation API
- [x] Configurer Sphinx avec autodoc
- [x] Générer documentation HTML
- [ ] Publier sur GitHub Pages (optionnel)

### 5.4 Guides utilisateur
- [x] Guide de démarrage rapide
- [x] FAQ / Troubleshooting
- [x] Exemples d'utilisation avancée

| Livrable | Effort estimé | Statut |
|----------|---------------|--------|
| CONTRIBUTING.md | 1h | ✅ |
| CODE_OF_CONDUCT.md | 0.5h | ✅ |
| CHANGELOG.md | 1h | ✅ |
| Sphinx setup | 2h | ✅ |
| Guides utilisateur | 2h | ✅ |
| **Total Phase 5** | **6.5h** | ✅ |

---

## Phase 6 : MLOps & Monitoring ✅

**Objectif** : Assurer la maintenabilité du modèle en production

### 6.1 Model Registry
- [x] Versionner les modèles avec métadonnées
- [x] Stocker : hyperparamètres, métriques, date, hash des données
- [x] Format : `models/registry/v1.0.0/`

```json
// models/registry/v1.0.0/metadata.json
{
  "version": "1.0.0",
  "created_at": "2024-01-23",
  "n_clusters": 4,
  "silhouette_score": 0.68,
  "n_samples": 96096,
  "data_hash": "abc123..."
}
```

### 6.2 Data & Model Drift Detection
- [x] Créer `src/monitoring/drift.py`
- [x] Implémenter ARI (Adjusted Rand Index) pour détecter le drift
- [x] Alertes si ARI < 0.8
- [x] Test de Kolmogorov-Smirnov pour le drift de données

### 6.3 Notebook de maintenance
- [x] Créer `notebooks/05_maintenance.ipynb`
- [x] Workflow de ré-entraînement
- [x] Comparaison ancien vs nouveau modèle
- [x] Simulation temporelle du drift

### 6.4 Scheduling
- [x] GitHub Actions scheduled workflow (`.github/workflows/maintenance.yml`)
- [x] Exécution mensuelle automatique
- [x] Réentraînement conditionnel si drift détecté

| Livrable | Effort estimé | Statut |
|----------|---------------|--------|
| Model registry | 3h | ✅ |
| Drift detection | 4h | ✅ |
| Notebook maintenance | 2h | ✅ |
| Scheduling | 2h | ✅ |
| Tests (23 tests) | 2h | ✅ |
| **Total Phase 6** | **13h** | ✅ |

---

## Phase 7 : Améliorations Dashboard

**Objectif** : Enrichir l'expérience utilisateur

### 7.1 Fonctionnalités
- [ ] Export CSV/Excel des données filtrées
- [ ] Comparaison côte-à-côte des segments
- [ ] Filtres avancés sur toutes les pages
- [ ] Graphiques de tendance temporelle

### 7.2 UX/UI
- [ ] Loading indicators pendant les calculs
- [ ] Messages d'erreur user-friendly
- [ ] Tooltips explicatifs
- [ ] Responsive design amélioré

### 7.3 Performance
- [ ] Optimiser le caching Streamlit
- [ ] Lazy loading des données volumineuses
- [ ] Pagination des tableaux

| Livrable | Effort estimé |
|----------|---------------|
| Export données | 2h |
| Comparaison segments | 3h |
| UX améliorations | 3h |
| Performance | 2h |
| **Total Phase 7** | **10h** |

---

## Calendrier suggéré

```
SEMAINE 1                    SEMAINE 2                    SEMAINE 3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
│   PHASE 1       │          │   PHASE 2       │          │   PHASE 3       │
│   Fondations    │   ───►   │   Tests &       │   ───►   │   CLI &         │
│   (~8h)         │          │   Robustesse    │          │   Automatisation│
│                 │          │   (~12h)        │          │   (~7h)         │
└─────────────────┘          └─────────────────┘          └─────────────────┘

SEMAINE 4                    SEMAINE 5                    SEMAINE 6+
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
│   PHASE 4       │          │   PHASE 5       │          │   PHASE 6 & 7   │
│   CI/CD         │   ───►   │   Documentation │   ───►   │   MLOps &       │
│   (~6h)         │          │   (~6h)         │          │   Dashboard     │
│                 │          │                 │          │   (~21h)        │
└─────────────────┘          └─────────────────┘          └─────────────────┘
```

---

## Résumé des efforts

| Phase | Description | Effort | Priorité |
|-------|-------------|--------|----------|
| 1 | Fondations & Qualité | 8h | 🔴 Critique |
| 2 | Tests & Robustesse | 12h | 🔴 Critique |
| 3 | CLI & Automatisation | 7h | 🟡 Important |
| 4 | CI/CD Complet | 6h | 🟡 Important |
| 5 | Documentation | 6h | 🟡 Important |
| 6 | MLOps & Monitoring | 11h | 🟢 Nice-to-have |
| 7 | Dashboard avancé | 10h | 🟢 Nice-to-have |
| **TOTAL** | | **60h** | |

---

## Métriques de succès

### Fin Phase 2
- [ ] Couverture tests > 80%
- [ ] 0 warning mypy/ruff
- [ ] Pre-commit hooks actifs

### Fin Phase 4
- [ ] Déploiement automatique fonctionnel
- [ ] Versioning sémantique en place
- [ ] < 5 min entre merge et deploy

### Fin Phase 7
- [ ] Silhouette score documenté et trackable
- [ ] Drift detection opérationnelle
- [ ] Dashboard avec toutes les fonctionnalités

---

## Dépendances entre phases

```
Phase 1 ─────► Phase 2 ─────► Phase 3
   │              │              │
   │              │              ▼
   │              │          Phase 4
   │              │              │
   │              ▼              ▼
   └────────► Phase 5 ◄─────────┘
                  │
                  ▼
              Phase 6
                  │
                  ▼
              Phase 7
```

**Notes** :
- Phase 1 est un prérequis pour toutes les autres
- Phases 3 et 4 peuvent être parallélisées
- Phases 6 et 7 sont indépendantes et optionnelles

---

## Pour commencer

```bash
# Cloner et setup
git clone <repo>
cd Projet_4
pip install -e ".[dev]"

# Lancer les tests actuels
pytest tests/ -v --cov=src

# Voir la couverture
open htmlcov/index.html
```

**Prochaine action** : Commencer par la Phase 1.1 (Refactoring des imports)

---

*Dernière mise à jour : 2024-01-23*
