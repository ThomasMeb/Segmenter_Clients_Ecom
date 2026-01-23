API Python
==========

Ce guide présente l'utilisation de l'API Python pour la segmentation client.

Concepts de base
----------------

Le workflow typique comprend trois étapes :

1. **Chargement des données** : Importer les transactions
2. **Feature Engineering** : Calculer les features RFM
3. **Segmentation** : Appliquer le clustering

Exemple complet
---------------

.. code-block:: python

   from src.data.loader import load_transactions
   from src.features.rfm import RFMCalculator
   from src.models.clustering import CustomerSegmenter
   from src.models.evaluation import evaluate_clustering

   # 1. Charger les données
   df = load_transactions("data/raw/data.csv")
   print(f"Transactions chargées: {len(df)}")

   # 2. Calculer les features RFM
   calculator = RFMCalculator(reference_date="2018-09-01")
   rfm = calculator.fit_transform(df)
   print(f"Clients uniques: {len(rfm)}")

   # 3. Segmenter les clients
   segmenter = CustomerSegmenter(n_clusters=4)
   labels = segmenter.fit_predict(rfm)

   # 4. Évaluer le modèle
   metrics = evaluate_clustering(rfm, labels)
   print(f"Score Silhouette: {metrics['silhouette_score']:.3f}")

   # 5. Obtenir le résumé par segment
   summary = segmenter.get_segment_summary(rfm)
   print(summary)

Chargement des données
----------------------

.. code-block:: python

   from src.data.loader import load_transactions

   # Charger depuis un fichier CSV
   df = load_transactions("chemin/vers/data.csv")

   # Le DataFrame contient les colonnes:
   # - customer_unique_id
   # - order_purchase_timestamp
   # - price
   # - order_id

Prétraitement
-------------

.. code-block:: python

   from src.data.preprocessor import preprocess_transactions

   # Nettoyer et préparer les données
   df_clean = preprocess_transactions(df)

   # Options de prétraitement:
   df_clean = preprocess_transactions(
       df,
       remove_duplicates=True,
       handle_missing=True,
       date_column="order_purchase_timestamp"
   )

Calcul RFM
----------

La classe ``RFMCalculator`` calcule les features Recency, Frequency et Monetary :

.. code-block:: python

   from src.features.rfm import RFMCalculator

   # Initialiser avec une date de référence
   calculator = RFMCalculator(reference_date="2018-09-01")

   # Calculer les features
   rfm = calculator.fit_transform(df)

   # Le résultat contient:
   # - recency: jours depuis le dernier achat
   # - frequency: nombre de commandes
   # - monetary: montant total dépensé

   # Obtenir les statistiques
   stats = calculator.get_statistics()
   print(stats)

   # Sauvegarder les résultats
   calculator.save("output/rfm.parquet", format="parquet")

Fonction utilitaire
^^^^^^^^^^^^^^^^^^^

Pour un usage rapide sans instancier la classe :

.. code-block:: python

   from src.features.rfm import calculate_rfm

   rfm = calculate_rfm(
       df,
       customer_col="customer_unique_id",
       date_col="order_purchase_timestamp",
       amount_col="price",
       reference_date="2018-09-01"
   )

Segmentation
------------

La classe ``CustomerSegmenter`` applique le clustering KMeans :

.. code-block:: python

   from src.models.clustering import CustomerSegmenter

   # Initialiser le segmenteur
   segmenter = CustomerSegmenter(
       n_clusters=4,
       random_state=42
   )

   # Entraîner et prédire
   labels = segmenter.fit_predict(rfm)

   # Ou en deux étapes
   segmenter.fit(rfm)
   labels = segmenter.predict(rfm)

   # Obtenir le résumé des segments
   summary = segmenter.get_segment_summary(rfm)

   # Sauvegarder le modèle
   segmenter.save("models/")

   # Charger un modèle existant
   segmenter = CustomerSegmenter.load("models/")

Évaluation
----------

.. code-block:: python

   from src.models.evaluation import (
       evaluate_clustering,
       calculate_silhouette_score,
       calculate_calinski_harabasz_score,
       find_optimal_clusters
   )

   # Évaluation complète
   metrics = evaluate_clustering(rfm, labels)
   print(f"Silhouette: {metrics['silhouette_score']:.3f}")
   print(f"Calinski-Harabasz: {metrics['calinski_harabasz_score']:.1f}")
   print(f"Davies-Bouldin: {metrics['davies_bouldin_score']:.3f}")

   # Trouver le nombre optimal de clusters
   optimal_k, scores = find_optimal_clusters(
       rfm,
       k_range=range(2, 10),
       method="silhouette"
   )
   print(f"Nombre optimal de clusters: {optimal_k}")

Visualisation
-------------

.. code-block:: python

   from src.visualization.plots import (
       plot_segment_distribution,
       plot_rfm_boxplots,
       plot_3d_clusters,
       plot_elbow_curve
   )

   # Distribution des segments
   fig = plot_segment_distribution(labels)
   fig.savefig("segments_distribution.png")

   # Boxplots RFM par segment
   fig = plot_rfm_boxplots(rfm, labels)
   fig.savefig("rfm_boxplots.png")

   # Visualisation 3D
   fig = plot_3d_clusters(rfm, labels)
   fig.show()

   # Courbe du coude
   fig = plot_elbow_curve(rfm, k_range=range(2, 10))
   fig.savefig("elbow_curve.png")

Configuration
-------------

Les paramètres du projet sont centralisés dans ``src/config.py`` :

.. code-block:: python

   from src.config import (
       N_CLUSTERS,
       RANDOM_STATE,
       SEGMENT_NAMES,
       SEGMENT_COLORS,
       SEGMENT_DESCRIPTIONS
   )

   # Personnaliser les noms de segments
   print(SEGMENT_NAMES)
   # {0: "Clients Récents", 1: "Clients Fidèles", ...}
