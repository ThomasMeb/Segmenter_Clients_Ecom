"""
Page Explorer - Exploration interactive des données.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Explorer", page_icon="🔍", layout="wide")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import SEGMENT_NAMES, SEGMENT_COLORS


@st.cache_data
def load_data():
    import sys
    root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(root))
    from app.utils import load_rfm_data
    return load_rfm_data()


def main():
    st.title("🔍 Exploration Interactive")

    data, is_real_data = load_data()

    if not is_real_data:
        st.info("📊 Mode Démonstration")

    # Filtres
    st.sidebar.subheader("Filtres")

    # Filtre récence
    recency_range = st.sidebar.slider(
        "Récence (jours)",
        min_value=int(data["recency"].min()),
        max_value=int(data["recency"].max()),
        value=(int(data["recency"].min()), int(data["recency"].max())),
    )

    # Filtre fréquence
    freq_range = st.sidebar.slider(
        "Fréquence",
        min_value=int(data["frequency"].min()),
        max_value=int(data["frequency"].max()),
        value=(int(data["frequency"].min()), int(data["frequency"].max())),
    )

    # Filtre montant
    monetary_range = st.sidebar.slider(
        "Montant (BRL)",
        min_value=float(data["monetary"].min()),
        max_value=float(data["monetary"].max()),
        value=(float(data["monetary"].min()), float(data["monetary"].max())),
    )

    # Filtre segment
    if "segment" in data.columns:
        segment_filter = st.sidebar.multiselect(
            "Segments",
            options=list(SEGMENT_NAMES.values()),
            default=list(SEGMENT_NAMES.values()),
        )
        selected_segments = [k for k, v in SEGMENT_NAMES.items() if v in segment_filter]
    else:
        selected_segments = None

    # Application des filtres
    filtered_data = data[
        (data["recency"] >= recency_range[0]) &
        (data["recency"] <= recency_range[1]) &
        (data["frequency"] >= freq_range[0]) &
        (data["frequency"] <= freq_range[1]) &
        (data["monetary"] >= monetary_range[0]) &
        (data["monetary"] <= monetary_range[1])
    ]

    if selected_segments is not None and "segment" in data.columns:
        filtered_data = filtered_data[filtered_data["segment"].isin(selected_segments)]

    # Stats filtrées
    st.info(f"**{len(filtered_data):,}** clients correspondent aux critères ({len(filtered_data)/len(data)*100:.1f}% du total)")

    st.divider()

    # Visualisations
    tab1, tab2, tab3 = st.tabs(["Scatter 2D", "Scatter 3D", "Données brutes"])

    with tab1:
        st.subheader("Scatter Plot 2D")

        col1, col2 = st.columns(2)

        with col1:
            x_axis = st.selectbox("Axe X", ["recency", "frequency", "monetary"], index=0)
        with col2:
            y_axis = st.selectbox("Axe Y", ["recency", "frequency", "monetary"], index=2)

        # Échantillonnage si trop de données
        sample_size = min(5000, len(filtered_data))
        plot_data = filtered_data.sample(sample_size) if len(filtered_data) > sample_size else filtered_data

        if "segment" in plot_data.columns:
            plot_data["segment_name"] = plot_data["segment"].map(SEGMENT_NAMES)

            fig = px.scatter(
                plot_data,
                x=x_axis,
                y=y_axis,
                color="segment_name",
                color_discrete_map={v: SEGMENT_COLORS[k] for k, v in SEGMENT_NAMES.items()},
                opacity=0.6,
                title=f"{y_axis.capitalize()} vs {x_axis.capitalize()}",
            )
        else:
            fig = px.scatter(
                plot_data,
                x=x_axis,
                y=y_axis,
                opacity=0.6,
                title=f"{y_axis.capitalize()} vs {x_axis.capitalize()}",
            )

        st.plotly_chart(fig, width='stretch')

    with tab2:
        st.subheader("Visualisation 3D")

        sample_size = min(3000, len(filtered_data))
        plot_data = filtered_data.sample(sample_size) if len(filtered_data) > sample_size else filtered_data

        if "segment" in plot_data.columns:
            plot_data["segment_name"] = plot_data["segment"].map(SEGMENT_NAMES)

            fig = px.scatter_3d(
                plot_data,
                x="recency",
                y="frequency",
                z="monetary",
                color="segment_name",
                color_discrete_map={v: SEGMENT_COLORS[k] for k, v in SEGMENT_NAMES.items()},
                opacity=0.6,
                title="Segmentation 3D (RFM)",
            )
        else:
            fig = px.scatter_3d(
                plot_data,
                x="recency",
                y="frequency",
                z="monetary",
                opacity=0.6,
            )

        st.plotly_chart(fig, width='stretch')

    with tab3:
        st.subheader("Données filtrées")

        st.dataframe(filtered_data, width='stretch')

        # Export
        csv = filtered_data.to_csv(index=True)
        st.download_button(
            label="📥 Télécharger en CSV",
            data=csv,
            file_name="filtered_customers.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
