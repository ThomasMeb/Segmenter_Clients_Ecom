"""
Streamlit Dashboard for Customer Segmentation Analysis.

This interactive dashboard allows users to:
- Explore customer segments
- Visualize RFM distributions
- Compare clusters
- Search for specific customers
- Download segment data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import yaml

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def load_config():
    """Load configuration."""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


@st.cache_data
def load_data():
    """Load clustered customer data."""
    config = load_config()
    data_path = Path(config['data']['processed_dir']) / 'rfm_clustered.csv'

    if not data_path.exists():
        st.error(f"Clustered data not found at {data_path}. Please run train.py first.")
        return None

    return pd.read_csv(data_path)


def main():
    """Main dashboard application."""

    # Header
    st.markdown('<p class="main-header">📊 Customer Segmentation Dashboard</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Load data
    data = load_data()
    config = load_config()

    if data is None:
        st.stop()

    # Sidebar
    st.sidebar.header("⚙️ Dashboard Controls")

    # Cluster filter
    clusters = sorted(data['Cluster'].unique())
    cluster_descriptions = config.get('cluster_descriptions', {})

    selected_clusters = st.sidebar.multiselect(
        "Select Clusters to Display",
        options=clusters,
        default=clusters,
        format_func=lambda x: f"Cluster {x}: {cluster_descriptions.get(x, 'Unknown')}"
    )

    # Filter data
    filtered_data = data[data['Cluster'].isin(selected_clusters)]

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview",
        "🔍 Cluster Analysis",
        "👥 Customer Explorer",
        "📊 RFM Analysis",
        "📥 Export Data"
    ])

    # TAB 1: Overview
    with tab1:
        st.header("Segmentation Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Customers", f"{len(data):,}")

        with col2:
            st.metric("Active Clusters", len(selected_clusters))

        with col3:
            avg_monetary = filtered_data['Monetary'].mean()
            st.metric("Avg. Customer Value", f"${avg_monetary:.2f}")

        with col4:
            avg_frequency = filtered_data['Frequency'].mean()
            st.metric("Avg. Purchase Frequency", f"{avg_frequency:.2f}")

        st.markdown("---")

        # Cluster distribution
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Cluster Distribution")
            cluster_counts = filtered_data['Cluster'].value_counts().sort_index()

            fig = px.pie(
                values=cluster_counts.values,
                names=[f"Cluster {c}" for c in cluster_counts.index],
                title="Customer Distribution by Cluster",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Cluster Sizes")
            cluster_stats = filtered_data.groupby('Cluster').size().reset_index(name='Count')
            cluster_stats['Percentage'] = (cluster_stats['Count'] / len(data) * 100).round(2)
            cluster_stats['Description'] = cluster_stats['Cluster'].map(cluster_descriptions)

            st.dataframe(
                cluster_stats,
                use_container_width=True,
                hide_index=True
            )

    # TAB 2: Cluster Analysis
    with tab2:
        st.header("Cluster Analysis")

        # 3D scatter plot
        st.subheader("3D RFM Visualization")

        fig = px.scatter_3d(
            filtered_data,
            x='Recency',
            y='Frequency',
            z='Monetary',
            color='Cluster',
            title='Customer Segments in 3D RFM Space',
            labels={
                'Recency': 'Recency (days)',
                'Frequency': 'Frequency',
                'Monetary': 'Monetary ($)'
            },
            opacity=0.7,
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Cluster profiles
        st.subheader("Cluster Profiles")

        cluster_profiles = filtered_data.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].agg(
            ['mean', 'median', 'std', 'min', 'max']
        ).round(2)

        for cluster in selected_clusters:
            with st.expander(f"Cluster {cluster}: {cluster_descriptions.get(cluster, 'Unknown')}"):
                cluster_data = filtered_data[filtered_data['Cluster'] == cluster]

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Size", f"{len(cluster_data):,} customers")
                    st.metric("Avg Recency", f"{cluster_data['Recency'].mean():.0f} days")

                with col2:
                    st.metric("Percentage", f"{len(cluster_data)/len(data)*100:.1f}%")
                    st.metric("Avg Frequency", f"{cluster_data['Frequency'].mean():.2f}")

                with col3:
                    st.metric("Avg Monetary", f"${cluster_data['Monetary'].mean():.2f}")
                    st.metric("Total Value", f"${cluster_data['Monetary'].sum():,.2f}")

                # RFM bar chart for this cluster
                fig = go.Figure(data=[
                    go.Bar(name='Mean', x=['Recency', 'Frequency', 'Monetary'],
                           y=[cluster_data['Recency'].mean(),
                              cluster_data['Frequency'].mean(),
                              cluster_data['Monetary'].mean()])
                ])
                fig.update_layout(title=f"Mean RFM Values - Cluster {cluster}", height=300)
                st.plotly_chart(fig, use_container_width=True)

    # TAB 3: Customer Explorer
    with tab3:
        st.header("Customer Explorer")

        # Search by customer ID
        st.subheader("Search Customer")

        if 'customer_unique_id' in filtered_data.columns:
            customer_id = st.text_input("Enter Customer ID")

            if customer_id:
                customer_data = filtered_data[filtered_data['customer_unique_id'].str.contains(customer_id, case=False, na=False)]

                if len(customer_data) > 0:
                    st.success(f"Found {len(customer_data)} customer(s)")
                    st.dataframe(customer_data, use_container_width=True)
                else:
                    st.warning("No customer found with that ID")

        st.markdown("---")

        # Top customers
        st.subheader("Top Customers by Value")

        top_n = st.slider("Number of top customers", 5, 50, 10)
        top_customers = filtered_data.nlargest(top_n, 'Monetary')

        fig = px.bar(
            top_customers,
            x=top_customers.index,
            y='Monetary',
            color='Cluster',
            title=f"Top {top_n} Customers by Monetary Value",
            labels={'Monetary': 'Total Spend ($)', 'index': 'Customer Rank'}
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(top_customers[['Recency', 'Frequency', 'Monetary', 'Cluster']], use_container_width=True)

    # TAB 4: RFM Analysis
    with tab4:
        st.header("RFM Distribution Analysis")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Recency Distribution")
            fig = px.histogram(
                filtered_data,
                x='Recency',
                color='Cluster',
                title='Days Since Last Purchase',
                nbins=50
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Frequency Distribution")
            fig = px.histogram(
                filtered_data,
                x='Frequency',
                color='Cluster',
                title='Number of Purchases',
                nbins=30
            )
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            st.subheader("Monetary Distribution")
            fig = px.histogram(
                filtered_data,
                x='Monetary',
                color='Cluster',
                title='Total Spend ($)',
                nbins=50
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # 2D scatter plots
        st.subheader("2D Feature Relationships")

        x_feature = st.selectbox("X-axis", ['Recency', 'Frequency', 'Monetary'], index=0)
        y_feature = st.selectbox("Y-axis", ['Recency', 'Frequency', 'Monetary'], index=2)

        fig = px.scatter(
            filtered_data,
            x=x_feature,
            y=y_feature,
            color='Cluster',
            title=f'{x_feature} vs {y_feature}',
            opacity=0.6,
            marginal_x='box',
            marginal_y='box'
        )
        st.plotly_chart(fig, use_container_width=True)

    # TAB 5: Export Data
    with tab5:
        st.header("Export Segmentation Data")

        st.write("Download filtered customer segments for further analysis.")

        # Export options
        export_cluster = st.multiselect(
            "Select clusters to export",
            options=clusters,
            default=selected_clusters,
            format_func=lambda x: f"Cluster {x}: {cluster_descriptions.get(x, 'Unknown')}"
        )

        export_data = data[data['Cluster'].isin(export_cluster)]

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Customers to Export", f"{len(export_data):,}")

        with col2:
            st.metric("Total Value", f"${export_data['Monetary'].sum():,.2f}")

        # Download button
        csv = export_data.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"customer_segments_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

        # Preview
        st.subheader("Data Preview")
        st.dataframe(export_data.head(100), use_container_width=True)


if __name__ == "__main__":
    main()
