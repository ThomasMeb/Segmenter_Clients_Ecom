"""
Page Overview - Vue d'ensemble des segments.
"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Overview", page_icon="📊", layout="wide")

from app.utils import (
    apply_custom_css,
    create_download_buttons,
    load_rfm_data,
)
from src.config import SEGMENT_COLORS, SEGMENT_NAMES


@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    """Charge les données RFM avec cache d'une heure."""
    return load_rfm_data()


def main():
    # Appliquer le CSS personnalisé
    apply_custom_css()

    st.title("📊 Vue d'ensemble des Segments")

    # Chargement avec spinner
    with st.spinner("Chargement des données..."):
        data, is_real_data = load_data()

    if not is_real_data:
        st.info(
            "📊 **Mode Démonstration** - Données générées pour illustration. "
            "Les vraies données seront utilisées si disponibles."
        )

    # KPIs avec tooltips
    st.subheader("Indicateurs clés")

    col1, col2, col3, col4 = st.columns(4)

    if "segment" in data.columns:
        total_clients = len(data)

        with col1:
            st.metric(
                "Total Clients",
                f"{total_clients:,}",
                help="Nombre total de clients dans la base",
            )

        with col2:
            loyal = data[data["segment"] == 1]
            pct_loyal = len(loyal) / total_clients * 100
            st.metric(
                "Clients Fidèles",
                f"{pct_loyal:.1f}%",
                help="Clients avec achats réguliers (segment 1)",
            )

        with col3:
            vip = data[data["segment"] == 3]
            pct_vip = len(vip) / total_clients * 100
            st.metric(
                "Clients VIP",
                f"{pct_vip:.1f}%",
                help="Clients à forte valeur (segment 3)",
            )

        with col4:
            dormant = data[data["segment"] == 2]
            pct_dormant = len(dormant) / total_clients * 100
            st.metric(
                "Clients Dormants",
                f"{pct_dormant:.1f}%",
                delta=f"-{pct_dormant:.0f}% à réactiver",
                delta_color="inverse",
                help="Clients inactifs nécessitant une réactivation (segment 2)",
            )

    st.divider()

    # Graphiques
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Distribution des segments")

        if "segment" in data.columns:
            segment_counts = data["segment"].value_counts().sort_index()

            fig = px.pie(
                values=segment_counts.values,
                names=[
                    SEGMENT_NAMES.get(i, f"Segment {i}") for i in segment_counts.index
                ],
                color_discrete_sequence=[
                    SEGMENT_COLORS.get(i, "#999999") for i in segment_counts.index
                ],
                hole=0.4,
            )
            fig.update_traces(
                textinfo="percent+label",
                hovertemplate="<b>%{label}</b><br>%{value:,} clients<br>%{percent}<extra></extra>",
            )
            fig.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=-0.2),
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Profil des segments (Radar)")

        if "segment" in data.columns:
            # Calcul des moyennes par segment
            segment_means = data.groupby("segment")[
                ["recency", "frequency", "monetary"]
            ].mean()

            # Normalisation
            segment_means_norm = (segment_means - segment_means.min()) / (
                segment_means.max() - segment_means.min() + 1e-10
            )

            fig = go.Figure()

            categories = ["Récence", "Fréquence", "Montant"]

            for segment_id in segment_means_norm.index:
                values = segment_means_norm.loc[segment_id].tolist()
                values.append(values[0])  # Fermer le polygone

                fig.add_trace(
                    go.Scatterpolar(
                        r=values,
                        theta=categories + [categories[0]],
                        fill="toself",
                        name=SEGMENT_NAMES.get(segment_id, f"Segment {segment_id}"),
                        line=dict(color=SEGMENT_COLORS.get(segment_id, "#999999")),
                    )
                )

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1]),
                ),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.3),
                margin=dict(t=20, b=20),
            )

            st.plotly_chart(fig, use_container_width=True)

            with st.expander("ℹ️ Comment lire ce graphique ?"):
                st.markdown(
                    """
                Le graphique radar permet de comparer les profils RFM des segments :
                - **Récence** : Plus le score est élevé, plus l'achat est récent
                - **Fréquence** : Plus le score est élevé, plus le client achète souvent
                - **Montant** : Plus le score est élevé, plus le client dépense

                Les valeurs sont normalisées entre 0 et 1 pour faciliter la comparaison.
                """
                )

    # Tableau récapitulatif
    st.divider()
    st.subheader("Résumé par segment")

    if "segment" in data.columns:
        summary = (
            data.groupby("segment")
            .agg(
                {
                    "recency": ["mean", "std"],
                    "frequency": ["mean", "std"],
                    "monetary": ["mean", "sum"],
                }
            )
            .round(2)
        )

        summary.columns = [
            "Récence (moy)",
            "Récence (std)",
            "Fréquence (moy)",
            "Fréquence (std)",
            "Montant (moy)",
            "CA Total",
        ]

        summary["Clients"] = data.groupby("segment").size()
        summary["% du total"] = (summary["Clients"] / total_clients * 100).round(1)
        summary.index = [SEGMENT_NAMES.get(i, f"Segment {i}") for i in summary.index]

        # Réorganiser les colonnes
        summary = summary[
            [
                "Clients",
                "% du total",
                "Récence (moy)",
                "Fréquence (moy)",
                "Montant (moy)",
                "CA Total",
            ]
        ]

        st.dataframe(
            summary.style.format(
                {
                    "Récence (moy)": "{:.0f} j",
                    "Fréquence (moy)": "{:.1f}",
                    "Montant (moy)": "{:.2f} BRL",
                    "CA Total": "{:,.0f} BRL",
                    "% du total": "{:.1f}%",
                }
            ),
            use_container_width=True,
        )

        # Export
        st.divider()
        st.subheader("📥 Exporter les données")

        export_tab1, export_tab2 = st.tabs(["Résumé par segment", "Données complètes"])

        with export_tab1:
            create_download_buttons(
                summary.reset_index(), "resume_segments", "overview_summary"
            )

        with export_tab2:
            st.caption(f"⚠️ Export de {len(data):,} lignes")
            create_download_buttons(data, "donnees_completes", "overview_full")


if __name__ == "__main__":
    main()
