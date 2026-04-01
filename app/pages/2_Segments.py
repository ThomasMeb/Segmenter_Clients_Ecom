"""
Page Segments - Analyse et comparaison des segments.
"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Segments", page_icon="👥", layout="wide")

from app.utils import (
    apply_custom_css,
    create_download_buttons,
    load_rfm_data,
    show_dataframe_with_pagination,
)
from src.config import SEGMENT_COLORS, SEGMENT_DESCRIPTIONS, SEGMENT_NAMES


@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    """Charge les données RFM avec cache."""
    return load_rfm_data()


def main():
    apply_custom_css()
    st.title("👥 Analyse des Segments")

    with st.spinner("Chargement des données..."):
        data, is_real_data = load_data()

    if "segment" not in data.columns:
        st.error("❌ Données de segmentation non disponibles.")
        return

    if not is_real_data:
        st.info("📊 **Mode Démonstration** - Données générées pour illustration.")

    # Mode d'affichage
    st.sidebar.subheader("Mode d'affichage")
    view_mode = st.sidebar.radio(
        "Choisir le mode",
        ["Analyse détaillée", "Comparaison côte-à-côte"],
        help="Sélectionnez le mode de visualisation des segments",
    )

    if view_mode == "Analyse détaillée":
        show_detailed_analysis(data)
    else:
        show_comparison_view(data)


def show_detailed_analysis(data):
    """Affiche l'analyse détaillée d'un segment."""
    # Sélecteur de segment
    st.sidebar.subheader("Sélection du segment")

    segment_options = {
        SEGMENT_NAMES.get(i, f"Segment {i}"): i
        for i in sorted(data["segment"].unique())
    }

    selected_name = st.sidebar.selectbox(
        "Choisir un segment",
        options=list(segment_options.keys()),
        help="Sélectionnez un segment pour voir son analyse détaillée",
    )

    selected_segment = segment_options[selected_name]
    segment_data = data[data["segment"] == selected_segment]
    segment_color = SEGMENT_COLORS.get(selected_segment, "#3498db")

    # Header du segment
    st.header(f"{selected_name}")
    st.info(SEGMENT_DESCRIPTIONS.get(selected_segment, ""))

    # Métriques
    col1, col2, col3, col4 = st.columns(4)
    total_clients = len(data)

    with col1:
        st.metric(
            "Nombre de clients",
            f"{len(segment_data):,}",
            help="Nombre total de clients dans ce segment",
        )

    with col2:
        pct = len(segment_data) / total_clients * 100
        st.metric(
            "% du total",
            f"{pct:.1f}%",
            help="Proportion de ce segment par rapport à l'ensemble",
        )

    with col3:
        st.metric(
            "Récence moyenne",
            f"{segment_data['recency'].mean():.0f} jours",
            help="Nombre moyen de jours depuis le dernier achat",
        )

    with col4:
        st.metric(
            "Panier moyen",
            f"{segment_data['monetary'].mean():.2f} BRL",
            help="Montant moyen dépensé par client",
        )

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
            color_discrete_sequence=[segment_color],
        )
        fig.update_layout(
            xaxis_title="Jours depuis dernier achat",
            yaxis_title="Nombre de clients",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_mid:
        fig = px.histogram(
            segment_data,
            x="frequency",
            nbins=20,
            title="Distribution de la fréquence",
            color_discrete_sequence=[segment_color],
        )
        fig.update_layout(
            xaxis_title="Nombre de commandes",
            yaxis_title="Nombre de clients",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        fig = px.histogram(
            segment_data,
            x="monetary",
            nbins=30,
            title="Distribution du montant",
            color_discrete_sequence=[segment_color],
        )
        fig.update_layout(
            xaxis_title="Montant total (BRL)",
            yaxis_title="Nombre de clients",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Recommandations marketing
    st.divider()
    st.subheader("💡 Recommandations Marketing")

    recommendations = {
        0: [
            ("🎁", "Envoyer une campagne de bienvenue personnalisée"),
            ("⭐", "Proposer des offres de fidélisation (programme de points)"),
            (
                "⏰",
                "Encourager un second achat avec une réduction limitée dans le temps",
            ),
        ],
        1: [
            ("👑", "Maintenir l'engagement avec un programme VIP"),
            (
                "🎯",
                "Offrir des avantages exclusifs (accès anticipé, livraison gratuite)",
            ),
            ("💬", "Solliciter des avis et témoignages"),
        ],
        2: [
            ("💌", 'Lancer une campagne de réactivation ("Vous nous manquez")'),
            ("🏷️", "Proposer une offre promotionnelle attractive"),
            ("📊", "Enquête de satisfaction pour comprendre l'inactivité"),
        ],
        3: [
            ("🌟", "Service client premium et personnalisé"),
            ("🎪", "Invitations à des événements exclusifs"),
            ("📦", "Offres sur mesure basées sur l'historique d'achat"),
        ],
    }

    for icon, rec in recommendations.get(selected_segment, []):
        st.markdown(f"{icon} {rec}")

    # Statistiques détaillées
    st.divider()
    st.subheader("📊 Statistiques détaillées")

    stats = segment_data[["recency", "frequency", "monetary"]].describe().T
    stats.columns = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    stats.index = ["Récence (jours)", "Fréquence", "Montant (BRL)"]

    st.dataframe(
        stats.style.format("{:.2f}"),
        use_container_width=True,
    )

    # Export et liste des clients
    st.divider()
    st.subheader("📋 Liste des clients du segment")

    show_dataframe_with_pagination(
        segment_data, page_size=20, key=f"segment_{selected_segment}"
    )

    st.divider()
    create_download_buttons(
        segment_data,
        f"segment_{selected_name.lower().replace(' ', '_')}",
        f"seg_{selected_segment}",
    )


def show_comparison_view(data):
    """Affiche la comparaison côte-à-côte des segments."""
    st.subheader("🔄 Comparaison des segments")

    # Sélection des segments à comparer
    st.sidebar.subheader("Segments à comparer")

    available_segments = sorted(data["segment"].unique())
    segment_names = [SEGMENT_NAMES.get(i, f"Segment {i}") for i in available_segments]

    col1, col2 = st.sidebar.columns(2)

    with col1:
        seg1_name = st.selectbox(
            "Segment 1",
            segment_names,
            index=0,
            key="compare_seg1",
        )
        seg1_id = available_segments[segment_names.index(seg1_name)]

    with col2:
        seg2_name = st.selectbox(
            "Segment 2",
            segment_names,
            index=min(1, len(segment_names) - 1),
            key="compare_seg2",
        )
        seg2_id = available_segments[segment_names.index(seg2_name)]

    # Données des deux segments
    seg1_data = data[data["segment"] == seg1_id]
    seg2_data = data[data["segment"] == seg2_id]

    # Comparaison des métriques
    st.markdown("### Comparaison des métriques")

    metrics_col1, metrics_col2 = st.columns(2)

    with metrics_col1:
        st.markdown(f"#### {seg1_name}")
        st.markdown(
            f"**Couleur:** <span style='color:{SEGMENT_COLORS.get(seg1_id, '#999')};'>●</span>",
            unsafe_allow_html=True,
        )
        st.metric("Clients", f"{len(seg1_data):,}")
        st.metric("Récence moy.", f"{seg1_data['recency'].mean():.0f} j")
        st.metric("Fréquence moy.", f"{seg1_data['frequency'].mean():.1f}")
        st.metric("Montant moy.", f"{seg1_data['monetary'].mean():.2f} BRL")
        st.metric("CA Total", f"{seg1_data['monetary'].sum():,.0f} BRL")

    with metrics_col2:
        st.markdown(f"#### {seg2_name}")
        st.markdown(
            f"**Couleur:** <span style='color:{SEGMENT_COLORS.get(seg2_id, '#999')};'>●</span>",
            unsafe_allow_html=True,
        )
        st.metric("Clients", f"{len(seg2_data):,}")
        st.metric("Récence moy.", f"{seg2_data['recency'].mean():.0f} j")
        st.metric("Fréquence moy.", f"{seg2_data['frequency'].mean():.1f}")
        st.metric("Montant moy.", f"{seg2_data['monetary'].mean():.2f} BRL")
        st.metric("CA Total", f"{seg2_data['monetary'].sum():,.0f} BRL")

    st.divider()

    # Graphiques comparatifs
    st.markdown("### Distributions comparées")

    comparison_data = data[data["segment"].isin([seg1_id, seg2_id])].copy()
    comparison_data["segment_name"] = comparison_data["segment"].map(SEGMENT_NAMES)

    tab1, tab2, tab3 = st.tabs(["Récence", "Fréquence", "Montant"])

    color_map = {
        seg1_name: SEGMENT_COLORS.get(seg1_id, "#3498db"),
        seg2_name: SEGMENT_COLORS.get(seg2_id, "#e74c3c"),
    }

    with tab1:
        fig = px.histogram(
            comparison_data,
            x="recency",
            color="segment_name",
            barmode="overlay",
            nbins=30,
            opacity=0.7,
            color_discrete_map=color_map,
            title="Comparaison de la distribution de récence",
        )
        fig.update_layout(
            xaxis_title="Jours depuis dernier achat", yaxis_title="Nombre de clients"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = px.histogram(
            comparison_data,
            x="frequency",
            color="segment_name",
            barmode="overlay",
            nbins=20,
            opacity=0.7,
            color_discrete_map=color_map,
            title="Comparaison de la distribution de fréquence",
        )
        fig.update_layout(
            xaxis_title="Nombre de commandes", yaxis_title="Nombre de clients"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = px.histogram(
            comparison_data,
            x="monetary",
            color="segment_name",
            barmode="overlay",
            nbins=30,
            opacity=0.7,
            color_discrete_map=color_map,
            title="Comparaison de la distribution de montant",
        )
        fig.update_layout(
            xaxis_title="Montant total (BRL)", yaxis_title="Nombre de clients"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Boxplots comparatifs
    st.divider()
    st.markdown("### Boxplots comparatifs")

    fig = go.Figure()

    for seg_id, seg_name in [(seg1_id, seg1_name), (seg2_id, seg2_name)]:
        seg_data = data[data["segment"] == seg_id]

        fig.add_trace(
            go.Box(
                y=seg_data["monetary"],
                name=f"{seg_name}",
                marker_color=SEGMENT_COLORS.get(seg_id, "#999999"),
                boxmean=True,
            )
        )

    fig.update_layout(
        title="Comparaison des montants (Boxplot)",
        yaxis_title="Montant (BRL)",
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tableau comparatif
    st.divider()
    st.markdown("### Tableau comparatif")

    comparison_df = pd.DataFrame(
        {
            "Métrique": [
                "Nombre de clients",
                "% du total",
                "Récence moyenne",
                "Fréquence moyenne",
                "Montant moyen",
                "CA Total",
            ],
            seg1_name: [
                f"{len(seg1_data):,}",
                f"{len(seg1_data) / len(data) * 100:.1f}%",
                f"{seg1_data['recency'].mean():.0f} jours",
                f"{seg1_data['frequency'].mean():.1f}",
                f"{seg1_data['monetary'].mean():.2f} BRL",
                f"{seg1_data['monetary'].sum():,.0f} BRL",
            ],
            seg2_name: [
                f"{len(seg2_data):,}",
                f"{len(seg2_data) / len(data) * 100:.1f}%",
                f"{seg2_data['recency'].mean():.0f} jours",
                f"{seg2_data['frequency'].mean():.1f}",
                f"{seg2_data['monetary'].mean():.2f} BRL",
                f"{seg2_data['monetary'].sum():,.0f} BRL",
            ],
        }
    )

    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # Export
    st.divider()
    create_download_buttons(comparison_data, "comparison_segments", "comparison")


# Import pandas for comparison_df
import pandas as pd

if __name__ == "__main__":
    main()
