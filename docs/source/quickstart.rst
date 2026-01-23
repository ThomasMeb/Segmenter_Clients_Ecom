Guide de démarrage rapide
=========================

Ce guide vous permettra d'être opérationnel en quelques minutes.

Prérequis
---------

* Python 3.10 ou supérieur
* Git
* Make (optionnel, mais recommandé)

Installation rapide
-------------------

.. code-block:: bash

   # Cloner le dépôt
   git clone https://github.com/ThomasMeb/olist-customer-segmentation.git
   cd olist-customer-segmentation

   # Exécuter le script de setup
   ./scripts/setup.sh

   # Ou manuellement :
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   pip install -e ".[dev]"

Premier entraînement
--------------------

Une fois installé, entraînez votre premier modèle :

.. code-block:: bash

   # Via CLI
   olist-segment train --input data/raw/data.csv --verbose

   # Ou via Make
   make train

Lancer le dashboard
-------------------

Visualisez vos segments dans le dashboard interactif :

.. code-block:: bash

   # Via CLI
   olist-segment serve

   # Ou via Make
   make serve

   # Ou directement avec Streamlit
   streamlit run app/app.py

Ouvrez ensuite http://localhost:8501 dans votre navigateur.

Utilisation Python
------------------

Vous pouvez également utiliser la bibliothèque directement en Python :

.. code-block:: python

   from src.data.loader import load_transactions
   from src.features.rfm import RFMCalculator
   from src.models.clustering import CustomerSegmenter

   # Charger les données
   df = load_transactions("data/raw/data.csv")

   # Calculer les features RFM
   calculator = RFMCalculator(reference_date="2018-09-01")
   rfm = calculator.fit_transform(df)

   # Segmenter les clients
   segmenter = CustomerSegmenter(n_clusters=4)
   labels = segmenter.fit_predict(rfm)

   # Obtenir le résumé
   summary = segmenter.get_segment_summary(rfm)
   print(summary)

Prochaines étapes
-----------------

* :doc:`usage/cli` - Découvrez toutes les commandes CLI
* :doc:`usage/api` - Référence complète de l'API Python
* :doc:`usage/dashboard` - Guide du dashboard Streamlit
* :doc:`faq` - Questions fréquentes
