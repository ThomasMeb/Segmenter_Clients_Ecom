Olist Customer Segmentation
============================

.. image:: https://img.shields.io/badge/Python-3.10%2B-blue.svg
   :target: https://www.python.org/

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT

Système de segmentation client utilisant l'analyse RFM et le clustering KMeans.

.. note::

   Ce projet a été initialement développé pour une mission client (nom confidentiel).
   Pour des raisons de portfolio, il a été adapté avec le dataset public Olist
   disponible sur `Kaggle <https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce>`_.

.. toctree::
   :maxdepth: 2
   :caption: Guide de démarrage

   quickstart
   installation

.. toctree::
   :maxdepth: 2
   :caption: Guide utilisateur

   usage/cli
   usage/api
   usage/dashboard

.. toctree::
   :maxdepth: 2
   :caption: Référence API

   api/data
   api/features
   api/models
   api/visualization

.. toctree::
   :maxdepth: 1
   :caption: Ressources

   faq
   changelog
   contributing

Aperçu du projet
----------------

Ce projet implémente une solution de segmentation client développée initialement pour une mission professionnelle (client confidentiel). En utilisant le framework RFM (Recency, Frequency, Monetary) et le machine learning non supervisé, nous identifions des segments de clients distincts pour permettre des stratégies marketing ciblées.

Pour la démonstration, la méthodologie est appliquée au dataset **Olist**, une marketplace e-commerce brésilienne dont les données sont publiquement disponibles sur Kaggle.

Fonctionnalités principales
---------------------------

* **Feature Engineering RFM** : Calcul automatisé de Recency, Frequency et Monetary
* **Clustering KMeans** : Segmentation optimisée avec analyse silhouette
* **Dashboard Interactif** : Visualisation Streamlit des segments
* **CLI complète** : Interface en ligne de commande pour toutes les opérations
* **Détection de Drift** : Simulation ARI pour la maintenance du modèle

Résultats clés
--------------

+-------------------------+--------+
| Métrique                | Valeur |
+=========================+========+
| Clients analysés        | 96,096 |
+-------------------------+--------+
| Segments identifiés     | 4      |
+-------------------------+--------+
| Score Silhouette        | 0.68   |
+-------------------------+--------+
| Fréquence de mise à jour| 3-4 mois|
+-------------------------+--------+

Indices et tables
-----------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
