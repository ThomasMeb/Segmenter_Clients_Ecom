"""
Page About - À propos du projet.
"""

import streamlit as st

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")


def main():
    st.title("ℹ️ À propos du projet")

    st.markdown("""
    ## Olist Customer Segmentation

    Ce projet de Data Science a pour objectif de segmenter les clients de la plateforme
    e-commerce brésilienne **Olist** en utilisant l'analyse **RFM** (Recency, Frequency, Monetary)
    et le clustering **KMeans**.

    ---

    ### 🎯 Objectif

    Identifier des groupes de clients homogènes pour permettre des actions marketing ciblées :
    - **Clients Récents** : Potentiel de fidélisation
    - **Clients Fidèles** : Programme de récompenses
    - **Clients Dormants** : Campagnes de réactivation
    - **Clients VIP** : Service premium

    ---

    ### 📊 Méthodologie

    1. **Collecte des données** : Dataset Olist (Kaggle)
    2. **Feature Engineering** : Calcul des features RFM
    3. **Preprocessing** : Standardisation (StandardScaler)
    4. **Modélisation** : KMeans (k=4)
    5. **Évaluation** : Silhouette Score = 0.677
    6. **Visualisation** : Dashboard Streamlit

    ---

    ### 🛠️ Stack Technique

    | Composant | Technologie |
    |-----------|-------------|
    | Langage | Python 3.10+ |
    | ML | Scikit-learn |
    | Data | Pandas, NumPy |
    | Visualisation | Plotly, Seaborn |
    | Dashboard | Streamlit |
    | CI/CD | GitHub Actions |

    ---

    ### 📁 Structure du projet

    ```
    olist-customer-segmentation/
    ├── src/               # Code source
    │   ├── data/          # Chargement & preprocessing
    │   ├── features/      # Calcul RFM
    │   ├── models/        # Clustering
    │   └── visualization/ # Graphiques
    ├── app/               # Dashboard Streamlit
    ├── notebooks/         # Jupyter notebooks
    ├── tests/             # Tests unitaires
    └── docs/              # Documentation
    ```

    ---

    ### 👨‍💻 Auteur

    **Thomas Mebarki**

    - 🔗 [LinkedIn](https://linkedin.com/in/thomasmebarki)
    - 🐙 [GitHub](https://github.com/thomasmebarki)

    ---

    ### 📚 Références

    - [Dataset Olist (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
    - [RFM Analysis](https://en.wikipedia.org/wiki/RFM_(market_research))
    - [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)

    ---

    ### 📄 License

    Ce projet est sous licence MIT.
    """)


if __name__ == "__main__":
    main()
