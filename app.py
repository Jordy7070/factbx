import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from typing import Tuple, Dict, Optional
import time

# Configuration de la page
st.set_page_config(
    page_title="Calcul des Tarifs de Transport",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration des types de données pour optimiser la mémoire
DTYPE_CONFIGS = {
    'commandes': {
        'Nom du partenaire': 'category',
        'Service de transport': 'category',
        'Pays destination': 'category',
        'Code Pays': 'category'
    },
    'tarifs': {
        'Partenaire': 'category',
        'Service': 'category',
        'Pays': 'category'
    }
}

# Initialisation du cache d'état pour stocker les données chargées et les résultats
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {
        'commandes': None,
        'tarifs': None,
        'prix_achat': None,
        'results': None,
        'files_uploaded': False,
        'calculation_done': False
    }

# Fonctions utilitaires
@st.cache_data(ttl=3600)
def load_data_optimized(file_uploader, dtype_config=None) -> pd.DataFrame:
    try:
        df = pd.read_excel(file_uploader, dtype=dtype_config, engine='openpyxl')
        return optimize_dataframe(df)
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {str(e)}")
        return pd.DataFrame()

def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Optimise la mémoire en convertissant les colonnes en types plus légers."""
    for col in df.columns:
        if df[col].dtype == 'object':
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
        elif df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

@st.cache_data
def calculer_tarifs_vectorise(commandes: pd.DataFrame, tarifs: pd.DataFrame, taxe_par_transporteur: dict) -> pd.DataFrame:
    """Calcule les tarifs de manière vectorisée pour optimiser les performances."""
    commandes['key'] = commandes.apply(lambda x: f"{x['Nom du partenaire']}_{x['Service de transport']}_{x['Pays destination']}", axis=1)
    tarifs['key'] = tarifs.apply(lambda x: f"{x['Partenaire']}_{x['Service']}_{x['Pays']}", axis=1)

    merged = pd.merge(commandes, tarifs, on='key', how='left')
    poids_mask = (merged['Poids expédition'] >= merged['PoidsMin']) & (merged['Poids expédition'] <= merged['PoidsMax'])
    merged.loc[~poids_mask, 'Prix'] = np.nan

    merged['Taxe'] = merged['Service de transport'].map(taxe_par_transporteur).fillna(0)
    result = pd.DataFrame({
        'Tarif de Base': merged['Prix'],
        'Taxe Gasoil': merged['Prix'] * (merged['Taxe'] / 100)
    })
    result['Tarif Total'] = result['Tarif de Base'] + result['Taxe Gasoil']
    return result

@st.cache_data
def application_calcul_tarif_optimise(commandes: pd.DataFrame, tarifs: pd.DataFrame, taxe_par_transporteur: dict, prix_achat_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    resultats = calculer_tarifs_vectorise(commandes, tarifs, taxe_par_transporteur)
    commandes[['Tarif de Base', 'Taxe Gasoil', 'Tarif Total']] = resultats

    commandes_tarifées = commandes[commandes['Tarif de Base'].notna()].copy()
    commandes_sans_tarif = commandes[commandes['Tarif de Base'].isna()].copy()

    if prix_achat_df is not None:
        commandes_tarifées = commandes_tarifées.merge(prix_achat_df, on=['Service de transport', 'Code Pays'], how="left")
        commandes_tarifées["Marge"] = commandes_tarifées["Tarif Total"] - commandes_tarifées["Prix d'Achat"]
        commandes_tarifées["Marge %"] = (commandes_tarifées["Marge"] / commandes_tarifées["Tarif Total"] * 100).round(2)

    return commandes_tarifées, commandes_sans_tarif

@st.cache_data
def create_summary_metrics(commandes_tarifées: pd.DataFrame) -> Dict:
    metrics = {
        "total_commandes": len(commandes_tarifées),
        "ca_total": commandes_tarifées["Tarif Total"].sum(),
        "marge_totale": commandes_tarifées["Marge"].sum() if "Marge" in commandes_tarifées else 0,
        "marge_moyenne": commandes_tarifées["Marge"].mean() if "Marge" in commandes_tarifées else 0,
        "marge_pourcentage": (commandes_tarifées["Marge"].sum() / commandes_tarifées["Tarif Total"].sum() * 100) if "Marge" in commandes_tarifées else 0
    }
    return metrics

@st.cache_data
def convertir_df_en_excel(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

def main():
    st.title("Application de Calcul des Tarifs de Transport")
    with st.sidebar:
        st.subheader("Paramètres de la Taxe Gasoil")
        taxe_par_transporteur = {transporteur: st.number_input(f"Taxe {transporteur} (%)", min_value=0.0, max_value=100.0, step=0.1) for transporteur in ["Colissimo", "Chronopost", "DHL", "UPS", "FedEx", "Mondial Relay"]}

    with st.expander("📁 Fichiers", expanded=not st.session_state.data_cache['files_uploaded']):
        col1, col2, col3 = st.columns(3)
        with col1:
            commandes_file = st.file_uploader("Fichier des commandes", type=['xlsx'])
        with col2:
            tarifs_file = st.file_uploader("Fichier des tarifs", type=['xlsx'])
        with col3:
            prix_achat_file = st.file_uploader("Fichier des prix d'achat (optionnel)", type=['xlsx'])

        if commandes_file and tarifs_file:
            try:
                st.session_state.data_cache = {
                    'commandes': load_data_optimized(commandes_file, DTYPE_CONFIGS['commandes']),
                    'tarifs': load_data_optimized(tarifs_file, DTYPE_CONFIGS['tarifs']),
                    'prix_achat': load_data_optimized(prix_achat_file) if prix_achat_file else None,
                    'files_uploaded': True,
                    'calculation_done': False,
                    'results': None
                }
                st.success("✅ Fichiers chargés avec succès!")
            except Exception as e:
                st.error(f"Erreur lors du chargement des fichiers : {str(e)}")

    if st.session_state.data_cache['files_uploaded'] and not st.session_state.data_cache['calculation_done']:
        if st.button("🚀 Lancer le calcul des tarifs"):
            try:
                with st.spinner("Calcul des tarifs en cours..."):
                    commandes_tarifées, commandes_sans_tarif = application_calcul_tarif_optimise(
                        st.session_state.data_cache['commandes'],
                        st.session_state.data_cache['tarifs'],
                        taxe_par_transporteur,
                        st.session_state.data_cache['prix_achat']
                    )
                    metrics = create_summary_metrics(commandes_tarifées)
                    st.session_state.data_cache['results'] = {'commandes_tarifées': commandes_tarifées, 'commandes_sans_tarif': commandes_sans_tarif, 'metrics': metrics}
                    st.session_state.data_cache['calculation_done'] = True
                    st.success("✨ Calculs terminés!")
            except Exception as e:
                st.error(f"Erreur lors du calcul des tarifs : {str(e)}")

    if st.session_state.data_cache['calculation_done']:
        results = st.session_state.data_cache['results']
        st.write("Résultats:", results)

if __name__ == "__main__":
    main()
