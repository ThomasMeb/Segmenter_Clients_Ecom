Dashboard Streamlit
===================

Le projet inclut un dashboard interactif pour visualiser et explorer les segments clients.

Lancement du dashboard
----------------------

.. code-block:: bash

   # Via CLI
   olist-segment serve

   # Via Make
   make serve

   # Directement avec Streamlit
   streamlit run app/app.py

Le dashboard sera accessible à http://localhost:8501

Pages du dashboard
------------------

Overview (Vue d'ensemble)
^^^^^^^^^^^^^^^^^^^^^^^^^

La page d'accueil présente :

* **Métriques clés** : Nombre de clients, segments, score silhouette
* **Distribution des segments** : Diagramme circulaire interactif
* **Statistiques RFM** : Moyenne par feature

Cette page permet d'avoir une vue rapide de l'état de la segmentation.

Segments
^^^^^^^^

Analyse détaillée de chaque segment :

* **Sélection du segment** : Menu déroulant pour choisir un segment
* **Profil du segment** : Caractéristiques moyennes RFM
* **Comparaison** : Graphiques de comparaison avec les autres segments
* **Liste des clients** : Tableau des clients du segment sélectionné

Explorer
^^^^^^^^

Visualisation interactive 3D :

* **Scatter 3D** : Visualisation des clusters dans l'espace RFM
* **Filtres** : Sélection des segments à afficher
* **Rotation** : Manipulation interactive du graphique
* **Export** : Téléchargement du graphique

About
^^^^^

Documentation du projet :

* **Méthodologie** : Explication de l'analyse RFM et du clustering
* **Segments** : Description de chaque segment
* **Maintenance** : Recommandations de mise à jour

Mode démo
---------

Si les données ne sont pas disponibles, le dashboard peut fonctionner en mode démo avec des données simulées :

.. code-block:: python

   # Le mode démo se déclenche automatiquement
   # si data/processed/customers_rfm.parquet n'existe pas

Les données de démo permettent d'explorer toutes les fonctionnalités du dashboard sans avoir besoin des vraies données.

Configuration
-------------

Le dashboard peut être configuré via :

Fichier ``.streamlit/config.toml``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: toml

   [theme]
   primaryColor = "#3498db"
   backgroundColor = "#FFFFFF"
   secondaryBackgroundColor = "#F0F2F6"

   [server]
   port = 8501
   headless = true

Variables d'environnement
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Port personnalisé
   STREAMLIT_SERVER_PORT=8080 streamlit run app/app.py

   # Mode headless (sans ouverture navigateur)
   STREAMLIT_SERVER_HEADLESS=true streamlit run app/app.py

Personnalisation
----------------

Structure des fichiers
^^^^^^^^^^^^^^^^^^^^^^

::

   app/
   ├── app.py              # Point d'entrée principal
   ├── utils.py            # Fonctions utilitaires
   └── pages/
       ├── 1_Overview.py   # Page Vue d'ensemble
       ├── 2_Segments.py   # Page Segments
       └── 3_Explorer.py   # Page Explorer

Ajouter une page
^^^^^^^^^^^^^^^^

Pour ajouter une nouvelle page, créez un fichier dans ``app/pages/`` :

.. code-block:: python

   # app/pages/4_MaPage.py
   import streamlit as st

   st.set_page_config(page_title="Ma Page", page_icon="📊")

   st.title("Ma nouvelle page")
   # Votre contenu ici

Streamlit numérote automatiquement les pages selon le préfixe du fichier.

Fonctions utilitaires
^^^^^^^^^^^^^^^^^^^^^

Le module ``app/utils.py`` fournit des fonctions pour :

* Chargement des données avec cache
* Génération de données de démo
* Formatage des métriques

.. code-block:: python

   from app.utils import load_data, generate_demo_data

   # Charger les données (avec cache Streamlit)
   rfm, labels = load_data()

   # Générer des données de démo
   rfm_demo, labels_demo = generate_demo_data(n_samples=1000)

Déploiement
-----------

Streamlit Cloud
^^^^^^^^^^^^^^^

1. Connectez votre dépôt GitHub à Streamlit Cloud
2. Configurez le fichier principal : ``app/app.py``
3. Le déploiement est automatique à chaque push

Docker
^^^^^^

.. code-block:: dockerfile

   FROM python:3.11-slim

   WORKDIR /app
   COPY . .

   RUN pip install -e .

   EXPOSE 8501

   CMD ["streamlit", "run", "app/app.py", "--server.port=8501"]

Construction et exécution :

.. code-block:: bash

   docker build -t olist-dashboard .
   docker run -p 8501:8501 olist-dashboard
