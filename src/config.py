"""
Configuration globale du projet.

Ce module centralise toutes les constantes et paramètres du projet
pour faciliter la maintenance et les modifications.
"""

from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

# =============================================================================
# DATA FILES
# =============================================================================

RAW_TRANSACTIONS_FILE = RAW_DATA_DIR / "data.csv"
RAW_TRANSACTIONS_WITH_REVIEW_FILE = RAW_DATA_DIR / "data_2.csv"
PROCESSED_RFM_FILE = PROCESSED_DATA_DIR / "customers_rfm.parquet"

# =============================================================================
# MODEL PARAMETERS
# =============================================================================

N_CLUSTERS = 4
RANDOM_STATE = 42
KMEANS_N_INIT = 10
KMEANS_MAX_ITER = 300

# =============================================================================
# FEATURE NAMES
# =============================================================================

CUSTOMER_ID_COL = "customer_unique_id"
ORDER_ID_COL = "order_id"
DATE_COL = "order_purchase_timestamp"
AMOUNT_COL = "price"

RFM_FEATURES = ["recency", "frequency", "monetary"]

# =============================================================================
# SEGMENT CONFIGURATION
# =============================================================================

SEGMENT_NAMES = {
    0: "Clients Récents",
    1: "Clients Fidèles",
    2: "Clients Dormants",
    3: "Clients VIP",
}

SEGMENT_COLORS = {
    0: "#3498db",  # Blue
    1: "#2ecc71",  # Green
    2: "#e74c3c",  # Red
    3: "#f39c12",  # Orange
}

SEGMENT_DESCRIPTIONS = {
    0: "Clients avec un achat récent mais peu fréquent. Potentiel de fidélisation.",
    1: "Clients réguliers avec plusieurs achats. Programme de fidélité recommandé.",
    2: "Clients inactifs depuis longtemps. Campagne de réactivation nécessaire.",
    3: "Clients à forte valeur. Traitement premium et rétention prioritaire.",
}

# =============================================================================
# VISUALIZATION
# =============================================================================

FIGURE_SIZE = (12, 8)
PLOT_STYLE = "whitegrid"
COLOR_PALETTE = "husl"

# =============================================================================
# MODEL FILES
# =============================================================================

KMEANS_MODEL_FILE = MODELS_DIR / "kmeans_model.pkl"
SCALER_FILE = MODELS_DIR / "scaler.pkl"
