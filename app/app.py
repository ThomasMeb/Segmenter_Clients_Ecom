"""
Point d'entrée du dashboard Streamlit.

Lance l'application avec : streamlit run app/app.py
"""

from pathlib import Path

import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="Olist Customer Segmentation",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.utils import load_rfm_data
from src.models.clustering import CustomerSegmenter

# Paths
ROOT_DIR = Path(__file__).parent.parent
MODELS_DIR = ROOT_DIR / "models"


@st.cache_data
def load_data():
    """Charge les données RFM avec mise en cache."""
    return load_rfm_data()


@st.cache_resource
def load_model():
    """Charge le modèle de segmentation."""
    try:
        return CustomerSegmenter.load(MODELS_DIR)
    except FileNotFoundError:
        return None


def main():
    """Fonction principale du dashboard."""

    # Header
    st.title("🛒 Olist Customer Segmentation")
    st.markdown(
        """
    Dashboard interactif pour explorer la segmentation des clients Olist
    basée sur l'analyse RFM (Recency, Frequency, Monetary).
    """
    )

    st.divider()

    # Chargement des données
    data, is_real_data = load_data()
    _model = load_model()  # Pour usage futur

    # Avertissement si données de démo
    if not is_real_data:
        st.info(
            """
        📊 **Mode Démonstration** - Les données affichées sont générées pour illustration.
        Pour utiliser les vraies données Olist, exécutez `python scripts/prepare_dashboard_data.py`.
        """
        )

    # Sidebar
    st.sidebar.title("Navigation")
    st.sidebar.markdown("Utilisez les pages pour explorer les différentes vues.")

    st.sidebar.divider()

    st.sidebar.markdown("### Statistiques rapides")
    st.sidebar.metric("Clients", f"{len(data):,}")

    if "segment" in data.columns:
        n_segments = data["segment"].nunique()
        st.sidebar.metric("Segments", n_segments)

    # Contenu principal
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Clients",
            value=f"{len(data):,}",
        )

    with col2:
        st.metric(
            label="Récence moyenne",
            value=f"{data['recency'].mean():.0f} jours",
        )

    with col3:
        st.metric(
            label="Fréquence moyenne",
            value=f"{data['frequency'].mean():.2f}",
        )

    with col4:
        st.metric(
            label="Panier moyen",
            value=f"{data['monetary'].mean():.2f} BRL",
        )

    st.divider()

    # Aperçu des données
    st.subheader("📊 Aperçu des données")
    st.dataframe(data.head(10), width="stretch")

    # Footer
    st.divider()
    st.markdown(
        """
    ---
    **Customer Segmentation** |
    [GitHub](https://github.com/ThomasMeb/olist-customer-segmentation) |
    Créé par Thomas Mebarki
    """
    )


if __name__ == "__main__":
    main()
