"""
Module de détection de drift pour les modèles de clustering.

Ce module fournit des outils pour détecter :
- Le drift de données (changement dans la distribution des features)
- Le drift de modèle (changement dans les assignations de clusters)
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score


@dataclass
class DriftReport:
    """
    Rapport de détection de drift.

    Attributes
    ----------
    timestamp : datetime
        Date du rapport.
    data_drift_detected : bool
        True si un drift de données est détecté.
    model_drift_detected : bool
        True si un drift de modèle est détecté.
    ari_score : float
        Score ARI entre les anciennes et nouvelles assignations.
    feature_drifts : dict
        Drift détecté par feature.
    recommendation : str
        Recommandation d'action.
    """

    timestamp: datetime
    data_drift_detected: bool
    model_drift_detected: bool
    ari_score: float | None
    feature_drifts: dict[str, dict]
    recommendation: str

    def to_dict(self) -> dict:
        """Convertit le rapport en dictionnaire."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "data_drift_detected": self.data_drift_detected,
            "model_drift_detected": self.model_drift_detected,
            "ari_score": self.ari_score,
            "feature_drifts": self.feature_drifts,
            "recommendation": self.recommendation,
        }


def calculate_ari(
    labels_reference: np.ndarray,
    labels_current: np.ndarray,
) -> float:
    """
    Calcule l'Adjusted Rand Index entre deux ensembles de labels.

    L'ARI mesure la similarité entre deux clusterings, ajusté pour
    la chance. Un score de 1 indique des clusterings identiques,
    0 indique un clustering aléatoire.

    Parameters
    ----------
    labels_reference : np.ndarray
        Labels de référence (ancien modèle).
    labels_current : np.ndarray
        Labels actuels (nouveau modèle ou nouvelles données).

    Returns
    -------
    float
        Score ARI entre -1 et 1.

    Examples
    --------
    >>> ari = calculate_ari(old_labels, new_labels)
    >>> if ari < 0.8:
    ...     print("Drift détecté!")
    """
    return adjusted_rand_score(labels_reference, labels_current)


def calculate_data_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    features: list[str] | None = None,
    threshold_pvalue: float = 0.05,
) -> dict[str, dict]:
    """
    Détecte le drift de données entre deux ensembles.

    Utilise le test de Kolmogorov-Smirnov pour comparer les
    distributions de chaque feature.

    Parameters
    ----------
    reference_data : pd.DataFrame
        Données de référence (entraînement).
    current_data : pd.DataFrame
        Données actuelles (production).
    features : list of str, optional
        Features à analyser. Si None, utilise toutes les colonnes.
    threshold_pvalue : float, default 0.05
        Seuil de p-value pour détecter un drift.

    Returns
    -------
    dict
        Dictionnaire {feature: {statistic, pvalue, drift_detected}}

    Examples
    --------
    >>> drift = calculate_data_drift(train_data, prod_data)
    >>> for feat, result in drift.items():
    ...     if result["drift_detected"]:
    ...         print(f"Drift détecté sur {feat}")
    """
    if features is None:
        features = list(reference_data.columns)

    results = {}

    for feature in features:
        if feature not in reference_data.columns or feature not in current_data.columns:
            continue

        ref_values = reference_data[feature].dropna().values
        cur_values = current_data[feature].dropna().values

        # Test de Kolmogorov-Smirnov
        statistic, pvalue = stats.ks_2samp(ref_values, cur_values)

        # Calcul de statistiques supplémentaires
        ref_mean = np.mean(ref_values)
        cur_mean = np.mean(cur_values)
        mean_shift = (cur_mean - ref_mean) / (ref_mean + 1e-10) * 100

        results[feature] = {
            "ks_statistic": float(statistic),
            "pvalue": float(pvalue),
            "drift_detected": pvalue < threshold_pvalue,
            "reference_mean": float(ref_mean),
            "current_mean": float(cur_mean),
            "mean_shift_percent": float(mean_shift),
        }

    return results


def check_model_drift(
    model: Any,
    scaler: Any,
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    ari_threshold: float = 0.8,
) -> tuple[bool, float]:
    """
    Vérifie si le modèle présente un drift.

    Compare les assignations du modèle sur les données de référence
    et les données actuelles (échantillon commun simulé).

    Parameters
    ----------
    model : Any
        Modèle de clustering (avec méthode predict).
    scaler : Any
        Scaler pour normaliser les données.
    reference_data : pd.DataFrame
        Données de référence.
    current_data : pd.DataFrame
        Données actuelles.
    ari_threshold : float, default 0.8
        Seuil ARI en dessous duquel on considère un drift.

    Returns
    -------
    tuple
        (drift_detected: bool, ari_score: float)
    """
    # Prédire sur les données de référence
    ref_scaled = scaler.transform(reference_data)
    ref_labels = model.predict(ref_scaled)

    # Prédire sur les données actuelles
    cur_scaled = scaler.transform(current_data)
    cur_labels = model.predict(cur_scaled)

    # Pour comparer, on utilise un échantillon de même taille
    min_size = min(len(ref_labels), len(cur_labels))

    # Ré-entraîner un modèle temporaire sur les nouvelles données
    # et comparer les assignations
    from sklearn.cluster import KMeans

    temp_model = KMeans(
        n_clusters=model.n_clusters,
        random_state=42,
        n_init=10,
    )
    temp_labels = temp_model.fit_predict(cur_scaled)

    # Calculer l'ARI entre les anciennes et nouvelles assignations
    # sur les données actuelles
    ari = calculate_ari(cur_labels[:min_size], temp_labels[:min_size])

    return ari < ari_threshold, ari


class DriftDetector:
    """
    Détecteur de drift pour les modèles de clustering.

    Cette classe combine la détection de drift de données et de modèle
    pour fournir un rapport complet et des recommandations.

    Attributes
    ----------
    ari_threshold : float
        Seuil ARI pour la détection de drift de modèle.
    pvalue_threshold : float
        Seuil p-value pour la détection de drift de données.
    reference_data : pd.DataFrame
        Données de référence stockées.
    reference_labels : np.ndarray
        Labels de référence stockés.

    Examples
    --------
    >>> detector = DriftDetector(ari_threshold=0.8)
    >>> detector.fit(train_data, train_labels)
    >>> report = detector.detect(new_data, model, scaler)
    >>> print(report.recommendation)
    """

    def __init__(
        self,
        ari_threshold: float = 0.8,
        pvalue_threshold: float = 0.05,
    ):
        """
        Initialise le détecteur de drift.

        Parameters
        ----------
        ari_threshold : float, default 0.8
            Seuil ARI pour considérer un drift de modèle.
        pvalue_threshold : float, default 0.05
            Seuil p-value pour considérer un drift de données.
        """
        self.ari_threshold = ari_threshold
        self.pvalue_threshold = pvalue_threshold
        self.reference_data: pd.DataFrame | None = None
        self.reference_labels: np.ndarray | None = None
        self._is_fitted = False

    def fit(
        self,
        reference_data: pd.DataFrame,
        reference_labels: np.ndarray,
    ) -> "DriftDetector":
        """
        Configure le détecteur avec les données de référence.

        Parameters
        ----------
        reference_data : pd.DataFrame
            Données d'entraînement de référence.
        reference_labels : np.ndarray
            Labels de clustering de référence.

        Returns
        -------
        DriftDetector
            Instance configurée.
        """
        self.reference_data = reference_data.copy()
        self.reference_labels = reference_labels.copy()
        self._is_fitted = True
        return self

    def detect(
        self,
        current_data: pd.DataFrame,
        model: Any,
        scaler: Any,
    ) -> DriftReport:
        """
        Détecte le drift sur les nouvelles données.

        Parameters
        ----------
        current_data : pd.DataFrame
            Nouvelles données à analyser.
        model : Any
            Modèle de clustering actuel.
        scaler : Any
            Scaler associé au modèle.

        Returns
        -------
        DriftReport
            Rapport complet de détection de drift.
        """
        if not self._is_fitted:
            raise ValueError("Le détecteur doit être configuré avec fit()")

        # Détecter le drift de données
        feature_drifts = calculate_data_drift(
            self.reference_data,
            current_data,
            threshold_pvalue=self.pvalue_threshold,
        )

        data_drift_detected = any(f["drift_detected"] for f in feature_drifts.values())

        # Détecter le drift de modèle
        model_drift_detected, ari_score = check_model_drift(
            model,
            scaler,
            self.reference_data,
            current_data,
            ari_threshold=self.ari_threshold,
        )

        # Générer la recommandation
        recommendation = self._generate_recommendation(
            data_drift_detected,
            model_drift_detected,
            ari_score,
            feature_drifts,
        )

        return DriftReport(
            timestamp=datetime.now(),
            data_drift_detected=data_drift_detected,
            model_drift_detected=model_drift_detected,
            ari_score=ari_score,
            feature_drifts=feature_drifts,
            recommendation=recommendation,
        )

    def _generate_recommendation(
        self,
        data_drift: bool,
        model_drift: bool,
        ari_score: float,
        feature_drifts: dict,
    ) -> str:
        """Génère une recommandation basée sur les résultats."""
        if model_drift and data_drift:
            drifted_features = [
                f for f, v in feature_drifts.items() if v["drift_detected"]
            ]
            return (
                f"URGENT: Drift de données ET de modèle détecté (ARI={ari_score:.3f}). "
                f"Features affectées: {', '.join(drifted_features)}. "
                "Réentraînement du modèle fortement recommandé."
            )
        elif model_drift:
            return (
                f"ATTENTION: Drift de modèle détecté (ARI={ari_score:.3f}). "
                "Les assignations de clusters ont significativement changé. "
                "Envisager un réentraînement du modèle."
            )
        elif data_drift:
            drifted_features = [
                f for f, v in feature_drifts.items() if v["drift_detected"]
            ]
            return (
                f"INFO: Drift de données détecté sur: {', '.join(drifted_features)}. "
                f"Le modèle reste stable (ARI={ari_score:.3f}). "
                "Surveiller l'évolution et planifier un réentraînement si le drift persiste."
            )
        else:
            return (
                f"OK: Aucun drift significatif détecté (ARI={ari_score:.3f}). "
                "Le modèle est stable. Prochain check recommandé dans 1 mois."
            )

    def simulate_temporal_drift(
        self,
        data: pd.DataFrame,
        model: Any,
        scaler: Any,
        n_periods: int = 12,
        drift_rate: float = 0.05,
    ) -> pd.DataFrame:
        """
        Simule le drift temporel pour estimer la fréquence de réentraînement.

        Parameters
        ----------
        data : pd.DataFrame
            Données de base.
        model : Any
            Modèle de clustering.
        scaler : Any
            Scaler associé.
        n_periods : int, default 12
            Nombre de périodes à simuler.
        drift_rate : float, default 0.05
            Taux de drift par période (5% = léger drift).

        Returns
        -------
        pd.DataFrame
            Évolution de l'ARI au fil du temps.
        """
        results = []
        base_data = data.copy()

        # Labels de référence
        base_scaled = scaler.transform(base_data)
        base_labels = model.predict(base_scaled)

        for period in range(n_periods):
            # Simuler le drift progressif
            drifted_data = base_data.copy()
            for col in drifted_data.columns:
                noise = np.random.normal(
                    0,
                    drifted_data[col].std() * drift_rate * (period + 1),
                    len(drifted_data),
                )
                drifted_data[col] = drifted_data[col] + noise

            # Prédire avec le modèle original
            drifted_scaled = scaler.transform(drifted_data)
            drifted_labels = model.predict(drifted_scaled)

            # Calculer l'ARI
            ari = calculate_ari(base_labels, drifted_labels)

            results.append(
                {
                    "period": period + 1,
                    "ari_score": ari,
                    "drift_accumulated": drift_rate * (period + 1) * 100,
                    "needs_retraining": ari < self.ari_threshold,
                }
            )

        return pd.DataFrame(results)


def estimate_retraining_frequency(
    data: pd.DataFrame,
    model: Any,
    scaler: Any,
    ari_threshold: float = 0.8,
    drift_rate: float = 0.05,
    max_periods: int = 24,
) -> dict:
    """
    Estime la fréquence optimale de réentraînement.

    Parameters
    ----------
    data : pd.DataFrame
        Données d'entraînement.
    model : Any
        Modèle entraîné.
    scaler : Any
        Scaler associé.
    ari_threshold : float, default 0.8
        Seuil ARI pour le réentraînement.
    drift_rate : float, default 0.05
        Taux de drift estimé par période.
    max_periods : int, default 24
        Nombre maximum de périodes à simuler.

    Returns
    -------
    dict
        Estimation avec period_until_drift et recommendation.
    """
    detector = DriftDetector(ari_threshold=ari_threshold)
    simulation = detector.simulate_temporal_drift(
        data, model, scaler, n_periods=max_periods, drift_rate=drift_rate
    )

    # Trouver la première période où le réentraînement est nécessaire
    needs_retraining = simulation[simulation["needs_retraining"]]

    if len(needs_retraining) > 0:
        first_retrain_period = needs_retraining.iloc[0]["period"]
        recommendation = (
            f"Réentraînement recommandé tous les {int(first_retrain_period)} mois "
            f"(basé sur un drift de {drift_rate*100:.1f}% par mois et ARI < {ari_threshold})"
        )
    else:
        first_retrain_period = max_periods
        recommendation = (
            f"Le modèle reste stable sur {max_periods} périodes. "
            "Réentraînement annuel suffisant."
        )

    return {
        "periods_until_drift": int(first_retrain_period),
        "ari_at_drift": float(
            simulation[simulation["period"] == first_retrain_period]["ari_score"].iloc[
                0
            ]
        ),
        "recommendation": recommendation,
        "simulation_data": simulation,
    }
