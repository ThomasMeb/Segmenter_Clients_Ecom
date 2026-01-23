Interface en ligne de commande (CLI)
====================================

Le projet fournit une CLI complète via la commande ``olist-segment``.

Commandes disponibles
---------------------

train
^^^^^

Entraîne le modèle de segmentation.

.. code-block:: bash

   olist-segment train [OPTIONS]

**Options :**

* ``--input, -i`` : Chemin vers le fichier de données (défaut: data/raw/data.csv)
* ``--output, -o`` : Dossier de sortie pour le modèle (défaut: models/)
* ``--n-clusters, -k`` : Nombre de clusters (défaut: 4)
* ``--verbose, -v`` : Mode verbeux

**Exemple :**

.. code-block:: bash

   olist-segment train --input data/raw/data.csv --n-clusters 5 --verbose

predict
^^^^^^^

Prédit les segments pour de nouveaux clients.

.. code-block:: bash

   olist-segment predict [OPTIONS]

**Options :**

* ``--input, -i`` : Fichier des nouveaux clients (requis)
* ``--output, -o`` : Fichier de sortie des prédictions
* ``--model, -m`` : Dossier du modèle (défaut: models/)
* ``--verbose, -v`` : Mode verbeux

**Exemple :**

.. code-block:: bash

   olist-segment predict --input new_customers.csv --output predictions.csv

evaluate
^^^^^^^^

Évalue les métriques du modèle.

.. code-block:: bash

   olist-segment evaluate [OPTIONS]

**Options :**

* ``--model, -m`` : Dossier du modèle (défaut: models/)
* ``--verbose, -v`` : Mode verbeux

**Exemple :**

.. code-block:: bash

   olist-segment evaluate --verbose

serve
^^^^^

Lance le dashboard Streamlit.

.. code-block:: bash

   olist-segment serve [OPTIONS]

**Options :**

* ``--port, -p`` : Port du serveur (défaut: 8501)
* ``--host`` : Adresse hôte (défaut: localhost)

**Exemple :**

.. code-block:: bash

   olist-segment serve --port 8080

info
^^^^

Affiche les informations du projet.

.. code-block:: bash

   olist-segment info [OPTIONS]

**Options :**

* ``--verbose, -v`` : Informations détaillées

**Exemple :**

.. code-block:: bash

   olist-segment info --verbose

Utilisation avec Make
---------------------

Le Makefile fournit des raccourcis pour les commandes courantes :

.. code-block:: bash

   # Entraîner le modèle
   make train

   # Évaluer le modèle
   make evaluate

   # Lancer le dashboard
   make serve

   # Voir toutes les commandes
   make help

Exemples de workflows
---------------------

Workflow complet d'entraînement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # 1. Télécharger les données
   ./scripts/download_data.sh

   # 2. Entraîner le modèle
   olist-segment train --verbose

   # 3. Évaluer les résultats
   olist-segment evaluate --verbose

   # 4. Lancer le dashboard
   olist-segment serve

Prédiction sur de nouveaux clients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Préparer un fichier avec les colonnes requises
   # customer_unique_id, order_purchase_timestamp, price, order_id

   # Prédire les segments
   olist-segment predict --input nouveaux_clients.csv --output resultats.csv
