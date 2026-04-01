"""
Utilitaires pour le dashboard Streamlit.

Ce module gère le chargement des données, la génération de données
de démonstration, et les fonctions d'export.
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


def get_project_root() -> Path:
    """Retourne le chemin racine du projet."""
    return Path(__file__).parent.parent


def generate_demo_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Génère des données RFM de démonstration.

    Utilisé quand les vraies données ne sont pas disponibles
    (ex: déploiement Streamlit Cloud sans Git LFS).

    Parameters
    ----------
    n_samples : int
        Nombre de clients à générer.
    random_state : int
        Graine aléatoire pour la reproductibilité.

    Returns
    -------
    pd.DataFrame
        DataFrame avec colonnes recency, frequency, monetary, segment.
    """
    np.random.seed(random_state)

    # Générer des données réalistes par segment
    segments = []

    # Segment 0 : Clients Récents (54%)
    n_recent = int(n_samples * 0.54)
    recent = pd.DataFrame(
        {
            "recency": np.random.randint(1, 90, n_recent),
            "frequency": np.ones(n_recent, dtype=int),
            "monetary": np.random.exponential(100, n_recent) + 20,
            "segment": 0,
        }
    )
    segments.append(recent)

    # Segment 1 : Clients Fidèles (3%)
    n_loyal = int(n_samples * 0.03)
    loyal = pd.DataFrame(
        {
            "recency": np.random.randint(10, 150, n_loyal),
            "frequency": np.random.randint(3, 10, n_loyal),
            "monetary": np.random.exponential(200, n_loyal) + 100,
            "segment": 1,
        }
    )
    segments.append(loyal)

    # Segment 2 : Clients Dormants (40%)
    n_dormant = int(n_samples * 0.40)
    dormant = pd.DataFrame(
        {
            "recency": np.random.randint(180, 400, n_dormant),
            "frequency": np.ones(n_dormant, dtype=int),
            "monetary": np.random.exponential(80, n_dormant) + 15,
            "segment": 2,
        }
    )
    segments.append(dormant)

    # Segment 3 : Clients VIP (3%)
    n_vip = n_samples - n_recent - n_loyal - n_dormant
    vip = pd.DataFrame(
        {
            "recency": np.random.randint(1, 60, n_vip),
            "frequency": np.random.randint(5, 15, n_vip),
            "monetary": np.random.exponential(500, n_vip) + 200,
            "segment": 3,
        }
    )
    segments.append(vip)

    # Combiner et mélanger
    df = pd.concat(segments, ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Ajouter un index client
    df.index = [f"customer_{i:05d}" for i in range(len(df))]
    df.index.name = "customer_unique_id"

    return df


def load_rfm_data() -> tuple[pd.DataFrame, bool]:
    """
    Charge les données RFM depuis le fichier ou génère des données de démo.

    Returns
    -------
    tuple[pd.DataFrame, bool]
        (DataFrame RFM, True si données réelles / False si démo)
    """
    root = get_project_root()

    # Essayer de charger les données réelles
    data_paths = [
        root / "data" / "processed" / "customers_rfm.parquet",
        root / "data" / "processed" / "customers_rfm.csv",
        root / "data" / "processed" / "data_RFM.csv",
    ]

    for path in data_paths:
        if path.exists():
            try:
                if path.suffix == ".parquet":
                    df = pd.read_parquet(path)
                else:
                    df = pd.read_csv(path, index_col=0)

                # Vérifier que les colonnes nécessaires sont présentes
                required_cols = ["recency", "frequency", "monetary"]
                if all(col in df.columns for col in required_cols):
                    return df, True
            except Exception:
                continue

    # Générer des données de démonstration
    return generate_demo_data(n_samples=5000), False


# =============================================================================
# FONCTIONS D'EXPORT
# =============================================================================


def convert_df_to_csv(df: pd.DataFrame) -> str:
    """Convertit un DataFrame en CSV string."""
    return df.to_csv(index=True)


def convert_df_to_excel(df: pd.DataFrame) -> bytes:
    """
    Convertit un DataFrame en fichier Excel.

    Returns
    -------
    bytes
        Contenu du fichier Excel.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=True, sheet_name="Données")
    return output.getvalue()


def create_download_buttons(
    df: pd.DataFrame,
    filename_base: str = "export",
    key_suffix: str = "",
) -> None:
    """
    Crée des boutons de téléchargement CSV et Excel.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à exporter.
    filename_base : str
        Nom de base du fichier (sans extension).
    key_suffix : str
        Suffixe pour les clés Streamlit (évite les duplications).
    """
    col1, col2 = st.columns(2)

    with col1:
        csv_data = convert_df_to_csv(df)
        st.download_button(
            label="📥 Télécharger CSV",
            data=csv_data,
            file_name=f"{filename_base}.csv",
            mime="text/csv",
            key=f"csv_{key_suffix}",
            help="Télécharger les données au format CSV",
        )

    with col2:
        try:
            excel_data = convert_df_to_excel(df)
            st.download_button(
                label="📥 Télécharger Excel",
                data=excel_data,
                file_name=f"{filename_base}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"excel_{key_suffix}",
                help="Télécharger les données au format Excel",
            )
        except ImportError:
            st.caption("📌 Installez openpyxl pour l'export Excel")


# =============================================================================
# FONCTIONS D'AFFICHAGE
# =============================================================================


def show_dataframe_with_pagination(
    df: pd.DataFrame,
    page_size: int = 20,
    key: str = "pagination",
) -> None:
    """
    Affiche un DataFrame avec pagination.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à afficher.
    page_size : int
        Nombre de lignes par page.
    key : str
        Clé unique pour le state Streamlit.
    """
    total_rows = len(df)
    total_pages = (total_rows - 1) // page_size + 1

    # Sélecteur de page
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        page = st.number_input(
            f"Page (1-{total_pages})",
            min_value=1,
            max_value=total_pages,
            value=1,
            key=f"{key}_page",
        )

    # Calculer les indices
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)

    # Afficher le sous-ensemble
    st.caption(f"Affichage des lignes {start_idx + 1} à {end_idx} sur {total_rows}")
    st.dataframe(df.iloc[start_idx:end_idx], use_container_width=True)


def format_number(value: float, decimals: int = 2) -> str:
    """Formate un nombre avec séparateurs de milliers."""
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.{decimals}f}M"
    elif abs(value) >= 1_000:
        return f"{value / 1_000:.{decimals}f}K"
    else:
        return f"{value:.{decimals}f}"


def create_metric_card(
    label: str,
    value: str | float,
    delta: str | float | None = None,
    help_text: str | None = None,
) -> None:
    """
    Crée une carte métrique avec style.

    Parameters
    ----------
    label : str
        Libellé de la métrique.
    value : str or float
        Valeur à afficher.
    delta : str or float, optional
        Variation par rapport à une référence.
    help_text : str, optional
        Texte d'aide au survol.
    """
    if isinstance(value, float):
        value = format_number(value)

    st.metric(label=label, value=value, delta=delta, help=help_text)


# =============================================================================
# FONCTIONS DE STYLE
# =============================================================================


def apply_custom_css() -> None:
    """Applique du CSS personnalisé au dashboard."""
    st.markdown(
        """
        <style>
        /* Amélioration des métriques */
        [data-testid="stMetricValue"] {
            font-size: 1.8rem;
            font-weight: 600;
        }

        /* Amélioration des tableaux */
        [data-testid="stDataFrame"] {
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        /* Style des boutons de téléchargement */
        .stDownloadButton > button {
            width: 100%;
        }

        /* Espacement des dividers */
        hr {
            margin: 1.5rem 0;
        }

        /* Info boxes */
        .stAlert {
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_loading_message(message: str = "Chargement en cours...") -> None:
    """Affiche un message de chargement."""
    st.info(f"⏳ {message}")
