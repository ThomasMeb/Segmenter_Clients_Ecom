"""
Model Registry pour le versioning des modèles.

Ce module permet de sauvegarder et charger des modèles avec leurs
métadonnées (version, métriques, hyperparamètres, hash des données).
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import MODELS_DIR


class ModelRegistry:
    """
    Registre de modèles avec versioning et métadonnées.

    Ce registre permet de :
    - Sauvegarder des modèles avec leurs métadonnées
    - Versionner automatiquement les modèles
    - Tracer les hyperparamètres et métriques
    - Calculer un hash des données d'entraînement

    Attributes
    ----------
    registry_dir : Path
        Dossier racine du registre.
    metadata_file : Path
        Fichier JSON contenant l'index des modèles.

    Examples
    --------
    >>> registry = ModelRegistry()
    >>> version = registry.register(
    ...     model=segmenter,
    ...     metrics={"silhouette": 0.68},
    ...     hyperparameters={"n_clusters": 4},
    ...     data_hash=registry.compute_data_hash(rfm_df)
    ... )
    >>> print(f"Modèle enregistré: v{version}")
    """

    def __init__(self, registry_dir: Path | str | None = None):
        """
        Initialise le registre de modèles.

        Parameters
        ----------
        registry_dir : Path or str, optional
            Dossier du registre. Par défaut: models/registry/
        """
        if registry_dir is None:
            self.registry_dir = MODELS_DIR / "registry"
        else:
            self.registry_dir = Path(registry_dir)

        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_dir / "registry.json"

        # Initialiser le fichier de métadonnées s'il n'existe pas
        if not self.metadata_file.exists():
            self._save_metadata({"models": [], "latest_version": "0.0.0"})

    def _load_metadata(self) -> dict:
        """Charge les métadonnées du registre."""
        with open(self.metadata_file) as f:
            return json.load(f)

    def _save_metadata(self, metadata: dict) -> None:
        """Sauvegarde les métadonnées du registre."""
        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    def _increment_version(self, current: str, bump: str = "patch") -> str:
        """
        Incrémente le numéro de version.

        Parameters
        ----------
        current : str
            Version actuelle (ex: "1.2.3")
        bump : str
            Type d'incrément: "major", "minor", ou "patch"

        Returns
        -------
        str
            Nouvelle version
        """
        parts = [int(x) for x in current.split(".")]

        if bump == "major":
            parts[0] += 1
            parts[1] = 0
            parts[2] = 0
        elif bump == "minor":
            parts[1] += 1
            parts[2] = 0
        else:  # patch
            parts[2] += 1

        return ".".join(str(x) for x in parts)

    @staticmethod
    def compute_data_hash(data: pd.DataFrame) -> str:
        """
        Calcule un hash SHA256 des données.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame à hasher.

        Returns
        -------
        str
            Hash SHA256 (16 premiers caractères).
        """
        # Convertir en bytes de manière reproductible
        data_bytes = pd.util.hash_pandas_object(data).values.tobytes()
        return hashlib.sha256(data_bytes).hexdigest()[:16]

    def register(
        self,
        model: Any,
        scaler: Any | None = None,
        metrics: dict[str, float] | None = None,
        hyperparameters: dict[str, Any] | None = None,
        data_hash: str | None = None,
        n_samples: int | None = None,
        description: str = "",
        bump: str = "patch",
    ) -> str:
        """
        Enregistre un nouveau modèle dans le registre.

        Parameters
        ----------
        model : Any
            Modèle à enregistrer (ex: KMeans, CustomerSegmenter).
        scaler : Any, optional
            Scaler associé au modèle.
        metrics : dict, optional
            Métriques d'évaluation (ex: {"silhouette": 0.68}).
        hyperparameters : dict, optional
            Hyperparamètres du modèle.
        data_hash : str, optional
            Hash des données d'entraînement.
        n_samples : int, optional
            Nombre d'échantillons d'entraînement.
        description : str, optional
            Description de cette version.
        bump : str, default "patch"
            Type d'incrément de version.

        Returns
        -------
        str
            Numéro de version enregistré.
        """
        metadata = self._load_metadata()

        # Calculer la nouvelle version
        new_version = self._increment_version(metadata["latest_version"], bump)

        # Créer le dossier de version
        version_dir = self.registry_dir / f"v{new_version}"
        version_dir.mkdir(parents=True, exist_ok=True)

        # Sauvegarder le modèle
        model_path = version_dir / "model.pkl"
        joblib.dump(model, model_path)

        # Sauvegarder le scaler si fourni
        if scaler is not None:
            scaler_path = version_dir / "scaler.pkl"
            joblib.dump(scaler, scaler_path)

        # Créer les métadonnées de cette version
        version_metadata = {
            "version": new_version,
            "created_at": datetime.now().isoformat(),
            "description": description,
            "metrics": metrics or {},
            "hyperparameters": hyperparameters or {},
            "data_hash": data_hash,
            "n_samples": n_samples,
            "model_path": str(model_path),
            "scaler_path": str(version_dir / "scaler.pkl") if scaler else None,
        }

        # Sauvegarder les métadonnées de version
        with open(version_dir / "metadata.json", "w") as f:
            json.dump(version_metadata, f, indent=2, default=str)

        # Mettre à jour le registre global
        metadata["models"].append(version_metadata)
        metadata["latest_version"] = new_version
        self._save_metadata(metadata)

        return new_version

    def load(self, version: str | None = None) -> tuple[Any, Any | None, dict]:
        """
        Charge un modèle depuis le registre.

        Parameters
        ----------
        version : str, optional
            Version à charger. Si None, charge la dernière.

        Returns
        -------
        tuple
            (model, scaler, metadata)
        """
        metadata = self._load_metadata()

        if version is None:
            version = metadata["latest_version"]

        version_dir = self.registry_dir / f"v{version}"

        if not version_dir.exists():
            raise ValueError(f"Version {version} non trouvée")

        # Charger les métadonnées
        with open(version_dir / "metadata.json") as f:
            version_metadata = json.load(f)

        # Charger le modèle
        model = joblib.load(version_dir / "model.pkl")

        # Charger le scaler si présent
        scaler = None
        scaler_path = version_dir / "scaler.pkl"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)

        return model, scaler, version_metadata

    def list_versions(self) -> pd.DataFrame:
        """
        Liste toutes les versions enregistrées.

        Returns
        -------
        pd.DataFrame
            Tableau des versions avec leurs métriques.
        """
        metadata = self._load_metadata()

        if not metadata["models"]:
            return pd.DataFrame()

        records = []
        for m in metadata["models"]:
            record = {
                "version": m["version"],
                "created_at": m["created_at"],
                "description": m["description"],
                "n_samples": m.get("n_samples"),
                "data_hash": m.get("data_hash"),
            }
            # Ajouter les métriques
            for key, value in m.get("metrics", {}).items():
                record[f"metric_{key}"] = value
            records.append(record)

        df = pd.DataFrame(records)
        df["created_at"] = pd.to_datetime(df["created_at"])
        return df.sort_values("version", ascending=False)

    def get_latest_version(self) -> str:
        """Retourne le numéro de la dernière version."""
        metadata = self._load_metadata()
        return metadata["latest_version"]

    def compare_versions(
        self,
        version1: str,
        version2: str,
    ) -> pd.DataFrame:
        """
        Compare deux versions du modèle.

        Parameters
        ----------
        version1 : str
            Première version.
        version2 : str
            Deuxième version.

        Returns
        -------
        pd.DataFrame
            Comparaison des métriques et hyperparamètres.
        """
        _, _, meta1 = self.load(version1)
        _, _, meta2 = self.load(version2)

        comparison = {
            "attribute": [],
            f"v{version1}": [],
            f"v{version2}": [],
        }

        # Comparer les métriques
        all_metrics = set(meta1.get("metrics", {}).keys()) | set(
            meta2.get("metrics", {}).keys()
        )
        for metric in sorted(all_metrics):
            comparison["attribute"].append(f"metric_{metric}")
            comparison[f"v{version1}"].append(meta1.get("metrics", {}).get(metric))
            comparison[f"v{version2}"].append(meta2.get("metrics", {}).get(metric))

        # Comparer les hyperparamètres
        all_params = set(meta1.get("hyperparameters", {}).keys()) | set(
            meta2.get("hyperparameters", {}).keys()
        )
        for param in sorted(all_params):
            comparison["attribute"].append(f"param_{param}")
            comparison[f"v{version1}"].append(
                meta1.get("hyperparameters", {}).get(param)
            )
            comparison[f"v{version2}"].append(
                meta2.get("hyperparameters", {}).get(param)
            )

        # Comparer les métadonnées
        for key in ["n_samples", "data_hash"]:
            comparison["attribute"].append(key)
            comparison[f"v{version1}"].append(meta1.get(key))
            comparison[f"v{version2}"].append(meta2.get(key))

        return pd.DataFrame(comparison)

    def delete_version(self, version: str) -> bool:
        """
        Supprime une version du registre.

        Parameters
        ----------
        version : str
            Version à supprimer.

        Returns
        -------
        bool
            True si supprimé avec succès.
        """
        import shutil

        version_dir = self.registry_dir / f"v{version}"

        if not version_dir.exists():
            return False

        # Supprimer le dossier
        shutil.rmtree(version_dir)

        # Mettre à jour les métadonnées
        metadata = self._load_metadata()
        metadata["models"] = [m for m in metadata["models"] if m["version"] != version]

        # Mettre à jour latest_version si nécessaire
        if metadata["latest_version"] == version and metadata["models"]:
            metadata["latest_version"] = max(m["version"] for m in metadata["models"])
        elif not metadata["models"]:
            metadata["latest_version"] = "0.0.0"

        self._save_metadata(metadata)
        return True
