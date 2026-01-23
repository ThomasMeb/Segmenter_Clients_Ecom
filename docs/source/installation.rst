Installation
============

Ce guide couvre les différentes méthodes d'installation du projet.

Prérequis système
-----------------

* **Python** : 3.10, 3.11 ou 3.12
* **Git** : Pour cloner le dépôt
* **Make** : Optionnel, pour les commandes automatisées
* **Git LFS** : Pour les fichiers de données volumineux

Installation standard
---------------------

Via le script de setup
^^^^^^^^^^^^^^^^^^^^^^

La méthode la plus simple :

.. code-block:: bash

   git clone https://github.com/ThomasMeb/olist-customer-segmentation.git
   cd olist-customer-segmentation
   ./scripts/setup.sh

Ce script effectue automatiquement :

1. Création de l'environnement virtuel
2. Installation des dépendances
3. Configuration des pre-commit hooks
4. Vérification de l'installation

Installation manuelle
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Cloner le dépôt
   git clone https://github.com/ThomasMeb/olist-customer-segmentation.git
   cd olist-customer-segmentation

   # Créer l'environnement virtuel
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # ou .venv\\Scripts\\activate  # Windows

   # Installer le package
   pip install -e .

Installation pour le développement
----------------------------------

Pour contribuer au projet, installez les dépendances de développement :

.. code-block:: bash

   pip install -e ".[dev]"
   pre-commit install

Ou via Make :

.. code-block:: bash

   make install-dev

Cela inclut :

* **pytest** : Framework de tests
* **pytest-cov** : Couverture de code
* **ruff** : Linter et formateur
* **mypy** : Vérification des types
* **pre-commit** : Hooks de pré-commit
* **sphinx** : Génération de documentation

Vérification de l'installation
------------------------------

Vérifiez que tout fonctionne correctement :

.. code-block:: bash

   # Vérifier la CLI
   olist-segment info

   # Lancer les tests
   make test

   # Vérifier le linting
   make lint

Téléchargement des données
--------------------------

Les données Olist peuvent être téléchargées depuis Kaggle :

.. code-block:: bash

   # Via le script fourni
   ./scripts/download_data.sh

   # Ou manuellement depuis Kaggle
   # https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

Dépannage
---------

Erreur "Module not found"
^^^^^^^^^^^^^^^^^^^^^^^^^

Assurez-vous d'avoir installé le package en mode éditable :

.. code-block:: bash

   pip install -e .

Erreur avec pre-commit
^^^^^^^^^^^^^^^^^^^^^^

Réinstallez les hooks :

.. code-block:: bash

   pre-commit uninstall
   pre-commit install

Problème avec les données
^^^^^^^^^^^^^^^^^^^^^^^^^

Vérifiez que les fichiers sont présents dans ``data/raw/`` :

.. code-block:: bash

   ls -la data/raw/
