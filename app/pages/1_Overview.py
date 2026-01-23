"""
Page Overview - Vue d'ensemble des segments.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Overview", page_icon="📊", layout="wide")

# Import configuration
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import SEGMENT_NAMES, SEGMENT_COLORS


@st.cache_data
def load_data():
    """Charge les données RFM."""
    import sys
    root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(root))
    from app.utils import load_rfm_data
    return load_rfm_data()


def main():
    st.title("📊 Vue d'ensemble des Segments")

    data, is_real_data = load_data()

    if not is_real_data:
        st.info("📊 Mode Démonstration - Données générées pour illustration.")

    # KPIs
    st.subheader("Indicateurs clés")

    col1, col2, col3, col4 = st.columns(4)

    if "segment" in data.columns:
        with col1:
            st.metric("Total Clients", f"{len(data):,}")

        with col2:
            loyal = data[data["segment"] == 1]
            pct_loyal = len(loyal) / len(data) * 100
            st.metric("Clients Fidèles", f"{pct_loyal:.1f}%")

        with col3:
            vip = data[data["segment"] == 3]
            pct_vip = len(vip) / len(data) * 100
            st.metric("Clients VIP", f"{pct_vip:.1f}%")

        with col4:
            dormant = data[data["segment"] == 2]
            pct_dormant = len(dormant) / len(data) * 100
            st.metric("Clients Dormants", f"{pct_dormant:.1f}%")

    st.divider()

    # Graphiques
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Distribution des segments")

        if "segment" in data.columns:
            segment_counts = data["segment"].value_counts().sort_index()

            fig = px.pie(
                values=segment_counts.values,
                names=[SEGMENT_NAMES.get(i, f"Segment {i}") for i in segment_counts.index],
                color_discrete_sequence=list(SEGMENT_COLORS.values()),
                hole=0.4,
            )
            fig.update_traces(textinfo="percent+label")
            st.plotly_chart(fig, width='stretch')

    with col_right:
        st.subheader("Profil des segments (Radar)")

        if "segment" in data.columns:
            # Calcul des moyennes par segment
            segment_means = data.groupby("segment")[["recency", "frequency", "monetary"]].mean()

            # Normalisation
            segment_means_norm = (segment_means - segment_means.min()) / (segment_means.max() - segment_means.min())

            fig = go.Figure()

            categories = ["Récence", "Fréquence", "Montant"]

            for segment_id in segment_means_norm.index:
                values = segment_means_norm.loc[segment_id].tolist()
                values.append(values[0])  # Fermer le polygone

                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],
                    fill="toself",
                    name=SEGMENT_NAMES.get(segment_id, f"Segment {segment_id}"),
                ))

            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
            )

            st.plotly_chart(fig, width='stretch')

    # Tableau récapitulatif
    st.divider()
    st.subheader("Résumé par segment")

    if "segment" in data.columns:
        summary = data.groupby("segment").agg({
            "recency": ["mean", "std"],
            "frequency": ["mean", "std"],
            "monetary": ["mean", "sum"],
        }).round(2)

        summary.columns = [
            "Récence (moy)", "Récence (std)",
            "Fréquence (moy)", "Fréquence (std)",
            "Montant (moy)", "CA Total",
        ]

        summary["Clients"] = data.groupby("segment").size()
        summary.index = [SEGMENT_NAMES.get(i, f"Segment {i}") for i in summary.index]

        st.dataframe(summary, width='stretch')


if __name__ == "__main__":
    main()
