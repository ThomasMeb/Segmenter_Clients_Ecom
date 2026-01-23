FAQ - Questions fréquentes
==========================

Cette page répond aux questions les plus courantes sur le projet.

Installation
------------

Comment installer le projet ?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

La méthode recommandée est d'utiliser le script de setup :

.. code-block:: bash

   git clone https://github.com/thomasmebarki/olist-customer-segmentation.git
   cd olist-customer-segmentation
   ./scripts/setup.sh

Voir :doc:`installation` pour plus de détails.

J'ai une erreur "ModuleNotFoundError"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Assurez-vous d'avoir installé le package en mode éditable :

.. code-block:: bash

   pip install -e .

Si l'erreur persiste, vérifiez que vous êtes bien dans l'environnement virtuel :

.. code-block:: bash

   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows

Quelle version de Python est requise ?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Le projet supporte Python 3.10, 3.11 et 3.12. Nous recommandons Python 3.11 pour les meilleures performances.

Données
-------

Où trouver les données Olist ?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Les données sont disponibles sur Kaggle :
`Brazilian E-Commerce Public Dataset by Olist <https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce>`_

Vous pouvez aussi utiliser le script de téléchargement :

.. code-block:: bash

   ./scripts/download_data.sh

Le dashboard fonctionne-t-il sans les données ?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Oui ! Le dashboard peut fonctionner en mode démo avec des données simulées. Ce mode se déclenche automatiquement si les données ne sont pas trouvées.

Comment utiliser mes propres données ?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Vos données doivent contenir les colonnes suivantes :

* ``customer_unique_id`` : Identifiant unique du client
* ``order_purchase_timestamp`` : Date de la commande
* ``price`` : Montant de la commande
* ``order_id`` : Identifiant de la commande

.. code-block:: python

   from src.data.loader import load_transactions

   # Charger vos données
   df = load_transactions("chemin/vers/vos_donnees.csv")

Modélisation
------------

Pourquoi 4 clusters ?
^^^^^^^^^^^^^^^^^^^^^

Le nombre de 4 clusters a été déterminé par :

1. **Méthode du coude** : Point d'inflexion de la courbe d'inertie
2. **Score silhouette** : Maximisation de la cohésion des clusters
3. **Interprétabilité business** : 4 segments distincts et actionnables

Vous pouvez expérimenter avec un nombre différent :

.. code-block:: bash

   olist-segment train --n-clusters 5

Comment interpréter le score silhouette ?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Le score silhouette varie de -1 à 1 :

* **> 0.7** : Structure forte
* **0.5 - 0.7** : Structure raisonnable (notre cas : 0.68)
* **0.25 - 0.5** : Structure faible
* **< 0.25** : Pas de structure claire

À quelle fréquence faut-il réentraîner le modèle ?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Nos analyses suggèrent un réentraînement tous les **3-4 mois**. Cela est basé sur :

* La simulation de drift avec l'Adjusted Rand Index (ARI)
* Le seuil d'alerte : ARI < 0.8

Dashboard
---------

Le dashboard ne s'affiche pas correctement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Vérifiez que Streamlit est bien installé :

   .. code-block:: bash

      pip install streamlit>=1.28.0

2. Essayez de vider le cache :

   .. code-block:: bash

      streamlit cache clear

3. Relancez avec le flag debug :

   .. code-block:: bash

      streamlit run app/app.py --logger.level=debug

Comment déployer le dashboard ?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Streamlit Cloud** (recommandé pour le prototypage) :

1. Poussez votre code sur GitHub
2. Connectez-vous à `share.streamlit.io <https://share.streamlit.io>`_
3. Déployez depuis votre dépôt

**Docker** (recommandé pour la production) :

.. code-block:: bash

   docker build -t olist-dashboard .
   docker run -p 8501:8501 olist-dashboard

Développement
-------------

Comment contribuer au projet ?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Forkez le dépôt
2. Créez une branche feature
3. Faites vos modifications
4. Soumettez une Pull Request

Voir :doc:`contributing` pour les détails.

Comment lancer les tests ?
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Tous les tests
   make test

   # Tests spécifiques
   pytest tests/test_rfm.py -v

   # Avec couverture
   pytest tests/ --cov=src --cov-report=html

Comment générer la documentation ?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cd docs
   make html
   open _build/html/index.html

Dépannage
---------

Erreur "CUDA out of memory"
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ce projet n'utilise pas CUDA par défaut. Si vous avez installé des dépendances GPU, assurez-vous d'avoir suffisamment de mémoire ou basculez sur CPU :

.. code-block:: python

   import os
   os.environ["CUDA_VISIBLE_DEVICES"] = ""

Les graphiques ne s'affichent pas
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pour les environnements sans interface graphique :

.. code-block:: python

   import matplotlib
   matplotlib.use('Agg')

Les tests échouent sur Windows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Certains chemins peuvent poser problème sur Windows. Utilisez ``pathlib`` :

.. code-block:: python

   from pathlib import Path
   data_path = Path("data") / "raw" / "data.csv"

Contact et support
------------------

* **Issues GitHub** : Pour les bugs et demandes de fonctionnalités
* **Discussions** : Pour les questions générales
* **Email** : thomas.mebarki@protonmail.com
