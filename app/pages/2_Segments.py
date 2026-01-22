"""
Page Segments - Détail de chaque segment.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Segments", page_icon="👥", layout="wide")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import SEGMENT_NAMES, SEGMENT_COLORS, SEGMENT_DESCRIPTIONS


@st.cache_data
def load_data():
    import sys
    root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(root))
    from app.utils import load_rfm_data
    return load_rfm_data()


def main():
    st.title("👥 Analyse des Segments")

    data, is_real_data = load_data()

    if "segment" not in data.columns:
        st.error("Données de segmentation non disponibles.")
        return

    if not is_real_data:
        st.info("📊 Mode Démonstration")

    # Sélecteur de segment
    st.sidebar.subheader("Sélection du segment")

    segment_options = {
        SEGMENT_NAMES.get(i, f"Segment {i}"): i
        for i in sorted(data["segment"].unique())
    }

    selected_name = st.sidebar.selectbox(
        "Choisir un segment",
        options=list(segment_options.keys()),
    )

    selected_segment = segment_options[selected_name]
    segment_data = data[data["segment"] == selected_segment]

    # Header du segment
    st.header(f"{selected_name}")
    st.info(SEGMENT_DESCRIPTIONS.get(selected_segment, ""))

    # Métriques
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Nombre de clients", f"{len(segment_data):,}")

    with col2:
        pct = len(segment_data) / len(data) * 100
        st.metric("% du total", f"{pct:.1f}%")

    with col3:
        st.metric("Récence moyenne", f"{segment_data['recency'].mean():.0f} jours")

    with col4:
        st.metric("Panier moyen", f"{segment_data['monetary'].mean():.2f} BRL")

    st.divider()

    # Distributions
    st.subheader("Distributions RFM")

    col_left, col_mid, col_right = st.columns(3)

    with col_left:
        fig = px.histogram(
            segment_data,
            x="recency",
            nbins=30,
            title="Distribution de la récence",
            color_discrete_sequence=[SEGMENT_COLORS.get(selected_segment, "#3498db")],
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_mid:
        fig = px.histogram(
            segment_data,
            x="frequency",
            nbins=20,
            title="Distribution de la fréquence",
            color_discrete_sequence=[SEGMENT_COLORS.get(selected_segment, "#3498db")],
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        fig = px.histogram(
            segment_data,
            x="monetary",
            nbins=30,
            title="Distribution du montant",
            color_discrete_sequence=[SEGMENT_COLORS.get(selected_segment, "#3498db")],
        )
        st.plotly_chart(fig, use_container_width=True)

    # Recommandations marketing
    st.divider()
    st.subheader("💡 Recommandations Marketing")

    recommendations = {
        0: [
            "Envoyer une campagne de bienvenue personnalisée",
            "Proposer des offres de fidélisation (programme de points)",
            "Encourager un second achat avec une réduction limitée dans le temps",
        ],
        1: [
            "Maintenir l'engagement avec un programme VIP",
            "Offrir des avantages exclusifs (accès anticipé, livraison gratuite)",
            "Solliciter des avis et témoignages",
        ],
        2: [
            "Lancer une campagne de réactivation (\"Vous nous manquez\")",
            "Proposer une offre promotionnelle attractive",
            "Enquête de satisfaction pour comprendre l'inactivité",
        ],
        3: [
            "Service client premium et personnalisé",
            "Invitations à des événements exclusifs",
            "Offres sur mesure basées sur l'historique d'achat",
        ],
    }

    for rec in recommendations.get(selected_segment, []):
        st.markdown(f"- {rec}")

    # Échantillon de clients
    st.divider()
    st.subheader("📋 Échantillon de clients")

    st.dataframe(
        segment_data.sample(min(10, len(segment_data))),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
