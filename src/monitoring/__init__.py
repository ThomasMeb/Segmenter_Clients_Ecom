"""
Module de monitoring MLOps.

Ce module fournit des outils pour :
- Le versioning des modèles avec métadonnées
- La détection de drift des données et du modèle
- Le suivi des performances en production
"""

from src.monitoring.drift import (
    DriftDetector,
    calculate_ari,
    calculate_data_drift,
    check_model_drift,
)
from src.monitoring.registry import ModelRegistry

__all__ = [
    "ModelRegistry",
    "DriftDetector",
    "calculate_ari",
    "calculate_data_drift",
    "check_model_drift",
]
