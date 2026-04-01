"""
Page Explorer - Exploration interactive des données.
"""

import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Explorer", page_icon="🔍", layout="wide")

from app.utils import (
    apply_custom_css,
    create_download_buttons,
    load_rfm_data,
    show_dataframe_with_pagination,
)
from src.config import SEGMENT_COLORS, SEGMENT_NAMES


@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    """Charge les données RFM avec cache."""
    return load_rfm_data()


def main():
    apply_custom_css()
    st.title("🔍 Exploration Interactive")

    with st.spinner("Chargement des données..."):
        data, is_real_data = load_data()

    if not is_real_data:
        st.info("📊 **Mode Démonstration** - Données générées pour illustration.")

    # Filtres dans la sidebar
    st.sidebar.subheader("🎛️ Filtres")

    with st.sidebar.expander("Récence", expanded=True):
        recency_range = st.slider(
            "Jours depuis dernier achat",
            min_value=int(data["recency"].min()),
            max_value=int(data["recency"].max()),
            value=(int(data["recency"].min()), int(data["recency"].max())),
            help="Filtrer par nombre de jours depuis le dernier achat",
        )

    with st.sidebar.expander("Fréquence", expanded=True):
        freq_range = st.slider(
            "Nombre de commandes",
            min_value=int(data["frequency"].min()),
            max_value=int(data["frequency"].max()),
            value=(int(data["frequency"].min()), int(data["frequency"].max())),
            help="Filtrer par nombre de commandes",
        )

    with st.sidebar.expander("Montant", expanded=True):
        monetary_range = st.slider(
            "Montant total (BRL)",
            min_value=float(data["monetary"].min()),
            max_value=float(data["monetary"].max()),
            value=(float(data["monetary"].min()), float(data["monetary"].max())),
            help="Filtrer par montant total dépensé",
        )

    # Filtre segment
    if "segment" in data.columns:
        with st.sidebar.expander("Segments", expanded=True):
            segment_filter = st.multiselect(
                "Sélectionner les segments",
                options=list(SEGMENT_NAMES.values()),
                default=list(SEGMENT_NAMES.values()),
                help="Filtrer par segment client",
            )
            selected_segments = [
                k for k, v in SEGMENT_NAMES.items() if v in segment_filter
            ]
    else:
        selected_segments = None

    # Bouton reset
    if st.sidebar.button("🔄 Réinitialiser les filtres", use_container_width=True):
        st.rerun()

    # Application des filtres
    filtered_data = data[
        (data["recency"] >= recency_range[0])
        & (data["recency"] <= recency_range[1])
        & (data["frequency"] >= freq_range[0])
        & (data["frequency"] <= freq_range[1])
        & (data["monetary"] >= monetary_range[0])
        & (data["monetary"] <= monetary_range[1])
    ]

    if selected_segments is not None and "segment" in data.columns:
        filtered_data = filtered_data[filtered_data["segment"].isin(selected_segments)]

    # Stats filtrées
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Clients filtrés",
            f"{len(filtered_data):,}",
            f"{len(filtered_data) / len(data) * 100:.1f}% du total",
            help="Nombre de clients correspondant aux critères",
        )

    with col2:
        st.metric(
            "Récence moyenne",
            f"{filtered_data['recency'].mean():.0f} jours",
            help="Récence moyenne des clients filtrés",
        )

    with col3:
        st.metric(
            "Montant moyen",
            f"{filtered_data['monetary'].mean():.2f} BRL",
            help="Montant moyen des clients filtrés",
        )

    st.divider()

    # Visualisations avec onglets
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Scatter 2D", "🧊 Scatter 3D", "📋 Données", "📈 Statistiques"]
    )

    with tab1:
        st.subheader("Scatter Plot 2D")

        col1, col2 = st.columns(2)

        with col1:
            x_axis = st.selectbox(
                "Axe X",
                ["recency", "frequency", "monetary"],
                index=0,
                format_func=lambda x: {
                    "recency": "Récence",
                    "frequency": "Fréquence",
                    "monetary": "Montant",
                }[x],
            )
        with col2:
            y_axis = st.selectbox(
                "Axe Y",
                ["recency", "frequency", "monetary"],
                index=2,
                format_func=lambda x: {
                    "recency": "Récence",
                    "frequency": "Fréquence",
                    "monetary": "Montant",
                }[x],
            )

        # Échantillonnage si trop de données
        sample_size = min(5000, len(filtered_data))
        if len(filtered_data) > sample_size:
            st.caption(
                f"⚠️ Échantillon de {sample_size:,} clients affiché pour performance"
            )
            plot_data = filtered_data.sample(sample_size, random_state=42)
        else:
            plot_data = filtered_data

        if "segment" in plot_data.columns:
            plot_data = plot_data.copy()
            plot_data["segment_name"] = plot_data["segment"].map(SEGMENT_NAMES)

            fig = px.scatter(
                plot_data,
                x=x_axis,
                y=y_axis,
                color="segment_name",
                color_discrete_map={
                    v: SEGMENT_COLORS[k] for k, v in SEGMENT_NAMES.items()
                },
                opacity=0.6,
                hover_data=["recency", "frequency", "monetary"],
            )
        else:
            fig = px.scatter(
                plot_data,
                x=x_axis,
                y=y_axis,
                opacity=0.6,
            )

        axis_labels = {
            "recency": "Récence (jours)",
            "frequency": "Fréquence",
            "monetary": "Montant (BRL)",
        }
        fig.update_layout(
            xaxis_title=axis_labels[x_axis],
            yaxis_title=axis_labels[y_axis],
            legend_title="Segment",
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Visualisation 3D")

        sample_size = min(3000, len(filtered_data))
        if len(filtered_data) > sample_size:
            st.caption(
                f"⚠️ Échantillon de {sample_size:,} clients affiché pour performance"
            )
            plot_data = filtered_data.sample(sample_size, random_state=42)
        else:
            plot_data = filtered_data

        if "segment" in plot_data.columns:
            plot_data = plot_data.copy()
            plot_data["segment_name"] = plot_data["segment"].map(SEGMENT_NAMES)

            fig = px.scatter_3d(
                plot_data,
                x="recency",
                y="frequency",
                z="monetary",
                color="segment_name",
                color_discrete_map={
                    v: SEGMENT_COLORS[k] for k, v in SEGMENT_NAMES.items()
                },
                opacity=0.6,
                hover_data=["recency", "frequency", "monetary"],
            )
        else:
            fig = px.scatter_3d(
                plot_data,
                x="recency",
                y="frequency",
                z="monetary",
                opacity=0.6,
            )

        fig.update_layout(
            scene=dict(
                xaxis_title="Récence (jours)",
                yaxis_title="Fréquence",
                zaxis_title="Montant (BRL)",
            ),
            legend_title="Segment",
        )

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("ℹ️ Comment utiliser le graphique 3D ?"):
            st.markdown(
                """
            - **Rotation** : Cliquez et faites glisser pour faire pivoter
            - **Zoom** : Utilisez la molette de la souris
            - **Pan** : Maintenez Shift + cliquez et faites glisser
            - **Reset** : Double-cliquez pour réinitialiser la vue
            """
            )

    with tab3:
        st.subheader("Données filtrées")

        # Options d'affichage
        col1, col2 = st.columns([3, 1])

        with col1:
            search_term = st.text_input(
                "🔍 Rechercher un client",
                placeholder="Entrez un ID client...",
                help="Rechercher par identifiant client",
            )

        with col2:
            sort_by = st.selectbox(
                "Trier par",
                ["recency", "frequency", "monetary"],
                format_func=lambda x: {
                    "recency": "Récence",
                    "frequency": "Fréquence",
                    "monetary": "Montant",
                }[x],
            )

        # Appliquer la recherche
        display_data = filtered_data.copy()
        if search_term:
            display_data = display_data[
                display_data.index.astype(str).str.contains(search_term, case=False)
            ]

        # Trier
        display_data = display_data.sort_values(sort_by, ascending=True)

        # Afficher avec pagination
        show_dataframe_with_pagination(display_data, page_size=25, key="explorer_data")

        # Export
        st.divider()
        st.subheader("📥 Exporter les données filtrées")
        create_download_buttons(filtered_data, "clients_filtres", "explorer")

    with tab4:
        st.subheader("Statistiques des données filtrées")

        # Statistiques descriptives
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Statistiques RFM")
            stats = filtered_data[["recency", "frequency", "monetary"]].describe()
            stats.index = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
            stats.columns = ["Récence", "Fréquence", "Montant"]
            st.dataframe(stats.style.format("{:.2f}"), use_container_width=True)

        with col2:
            if "segment" in filtered_data.columns:
                st.markdown("#### Distribution par segment")
                segment_dist = filtered_data["segment"].value_counts().sort_index()
                segment_dist.index = [
                    SEGMENT_NAMES.get(i, f"Segment {i}") for i in segment_dist.index
                ]

                fig = px.bar(
                    x=segment_dist.index,
                    y=segment_dist.values,
                    color=segment_dist.index,
                    color_discrete_map={
                        v: SEGMENT_COLORS.get(k, "#999")
                        for k, v in SEGMENT_NAMES.items()
                    },
                )
                fig.update_layout(
                    xaxis_title="Segment",
                    yaxis_title="Nombre de clients",
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

        # Corrélations
        st.markdown("#### Corrélations")

        corr = filtered_data[["recency", "frequency", "monetary"]].corr()
        corr.columns = ["Récence", "Fréquence", "Montant"]
        corr.index = ["Récence", "Fréquence", "Montant"]

        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
        )
        fig.update_layout(
            title="Matrice de corrélation",
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
