import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from typing import Tuple, Dict, Optional, List
import time

# Configuration des types de données
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

# Configuration de la page
st.set_page_config(
    page_title="Calcul des Tarifs de Transport",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialisation du state
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {
        'commandes': None,
        'tarifs': None,
        'prix_achat': None,
        'results': None,
        'files_uploaded': False,
        'calculation_done': False
    }

# Fonctions utilitaires optimisées
@st.cache_data(ttl=3600)
def load_data_optimized(file_uploader, dtype_config=None) -> pd.DataFrame:
    """Charge les données avec optimisation des types."""
    if dtype_config is None:
        dtype_config = {}
    
    try:
        df = pd.read_excel(
            file_uploader,
            dtype=dtype_config,
            engine='openpyxl'
        )
        return optimize_dataframe(df)
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier: {str(e)}")
        return pd.DataFrame()

def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Optimise la mémoire utilisée par le DataFrame."""
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
def calculer_tarifs_vectorise(commandes: pd.DataFrame, 
                            tarifs: pd.DataFrame, 
                            taxe_par_transporteur: dict) -> pd.DataFrame:
    """Calcule les tarifs de manière vectorisée."""
    # Préparation des données
    commandes = commandes.copy()
    tarifs = tarifs.copy()
    
    # Création d'une clé de fusion optimisée
    tarifs['key'] = tarifs.apply(
        lambda x: f"{x['Partenaire']}_{x['Service']}_{x['Pays']}", 
        axis=1
    )
    commandes['key'] = commandes.apply(
        lambda x: f"{x['Nom du partenaire']}_{x['Service de transport']}_{x['Pays destination']}", 
        axis=1
    )
    
    # Fusion optimisée
    merged = pd.merge(
        commandes,
        tarifs,
        on='key',
        how='left'
    )
    
    # Application vectorisée des conditions
    mask_poids = (
        (merged['Poids expédition'] >= merged['PoidsMin']) & 
        (merged['Poids expédition'] <= merged['PoidsMax'])
    )
    
    merged.loc[~mask_poids, 'Prix'] = np.nan
    
    # Gestion des doublons
    merged = merged.sort_values('Prix').groupby(merged.index).first()
    
    # Calcul vectorisé de la taxe
    merged['Taxe'] = 0
    for transporteur, taux in taxe_par_transporteur.items():
        mask_transporteur = merged['Service de transport'].str.lower().str.contains(
            transporteur.lower(), 
            regex=False
        )
        merged.loc[mask_transporteur, 'Taxe'] = taux
    
    # Calculs finaux vectorisés
    result = pd.DataFrame({
        'Tarif de Base': merged['Prix'],
        'Taxe Gasoil': merged['Prix'] * (merged['Taxe'] / 100)
    })
    result['Tarif Total'] = result['Tarif de Base'] + result['Taxe Gasoil']
    
    return result

@st.cache_data
def application_calcul_tarif_optimise(
    commandes: pd.DataFrame,
    tarifs: pd.DataFrame,
    taxe_par_transporteur: dict,
    prix_achat_df: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calcule les tarifs avec optimisation."""
    # Pré-traitement
    commandes = optimize_dataframe(commandes.copy())
    tarifs = optimize_dataframe(tarifs.copy())
    
    # Conversion numérique
    commandes['Poids expédition'] = pd.to_numeric(commandes['Poids expédition'], errors='coerce')
    tarifs['Prix'] = pd.to_numeric(tarifs['Prix'], errors='coerce')
    
    # Calcul des tarifs
    resultats = calculer_tarifs_vectorise(commandes, tarifs, taxe_par_transporteur)
    commandes[['Tarif de Base', 'Taxe Gasoil', 'Tarif Total']] = resultats
    
    # Séparation des résultats
    mask_tarif_trouve = commandes['Tarif de Base'].notna()
    commandes_tarifées = commandes[mask_tarif_trouve].copy()
    commandes_sans_tarif = commandes[~mask_tarif_trouve].copy()
    
    # Calcul des marges
    if prix_achat_df is not None:
        merge_cols = ["Service de transport"]
        if "Code Pays" in prix_achat_df.columns and "Code Pays" in commandes_tarifées.columns:
            merge_cols.append("Code Pays")
            
        commandes_tarifées = commandes_tarifées.merge(
            prix_achat_df,
            on=merge_cols,
            how="left"
        ).rename(columns={"Prix Achat": "Prix d'Achat"})
        
        commandes_tarifées["Marge"] = commandes_tarifées["Tarif Total"] - commandes_tarifées["Prix d'Achat"]
        
        # Calcul des métriques supplémentaires
        commandes_tarifées["Marge %"] = (commandes_tarifées["Marge"] / commandes_tarifées["Tarif Total"] * 100).round(2)
    
    return commandes_tarifées, commandes_sans_tarif

@st.cache_data
def create_summary_metrics(commandes_tarifées: pd.DataFrame) -> Dict:
    """Crée un résumé des métriques importantes."""
    return {
        "total_commandes": len(commandes_tarifées),
        "ca_total": commandes_tarifées["Tarif Total"].sum(),
        "marge_totale": commandes_tarifées["Marge"].sum() if "Marge" in commandes_tarifées else 0,
        "marge_moyenne": commandes_tarifées["Marge"].mean() if "Marge" in commandes_tarifées else 0,
        "marge_pourcentage": (commandes_tarifées["Marge"].sum() / commandes_tarifées["Tarif Total"].sum() * 100) if "Marge" in commandes_tarifées else 0
    }

def create_charts(commandes_tarifées: pd.DataFrame) -> Dict:
    """Crée les visualisations."""
    charts = {}
    
    # Graphique des tarifs par partenaire
    charts["tarifs_partenaire"] = px.bar(
        commandes_tarifées.groupby("Nom du partenaire")["Tarif Total"].sum().reset_index(),
        x="Nom du partenaire",
        y="Tarif Total",
        title="Tarif total par partenaire",
        labels={"Tarif Total": "Montant (€)", "Nom du partenaire": "Partenaire"}
    )
    
    if "Marge" in commandes_tarifées.columns:
        # Graphique des marges
        charts["marges"] = px.scatter(
            commandes_tarifées,
            x="Tarif Total",
            y="Marge",
            color="Service de transport",
            title="Marge vs Tarif",
            labels={"Tarif Total": "Tarif (€)", "Marge": "Marge (€)"}
        )
        
        # Graphique des marges par service
        charts["marges_service"] = px.box(
            commandes_tarifées,
            x="Service de transport",
            y="Marge %",
            title="Distribution des marges par service",
            labels={"Marge %": "Marge (%)", "Service de transport": "Service"}
        )
    
    return charts

def create_analysis_tabs(
    commandes_tarifées: pd.DataFrame,
    commandes_sans_tarif: pd.DataFrame,
    metrics: Dict,
    charts: Dict
):
    """Crée les onglets d'analyse."""
    tabs = st.tabs([
        "📊 Tableau de bord",
        "📈 Analyses détaillées",
        "📑 Données",
        "❌ Sans tarifs",
        "💾 Export"
    ])
    
    # Tableau de bord
    with tabs[0]:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total commandes", metrics["total_commandes"])
            st.metric("CA Total", f"{metrics['ca_total']:,.2f} €")
        with col2:
            st.metric("Marge totale", f"{metrics['marge_totale']:,.2f} €")
            st.metric("Marge moyenne", f"{metrics['marge_moyenne']:,.2f} €")
        with col3:
            st.metric("Marge globale", f"{metrics['marge_pourcentage']:.2f}%")
            st.metric("Commandes sans tarif", len(commandes_sans_tarif))
        
        st.plotly_chart(charts["tarifs_partenaire"], use_container_width=True)
        if "marges" in charts:
            st.plotly_chart(charts["marges"], use_container_width=True)
    
    # Analyses détaillées
    with tabs[1]:
        if "marges_service" in charts:
            st.plotly_chart(charts["marges_service"], use_container_width=True)
        
        # Analyses supplémentaires par pays, service, etc.
        analyse_type = st.selectbox(
            "Type d'analyse",
            ["Par pays", "Par service", "Par partenaire"]
        )
        
        if analyse_type == "Par pays":
            groupby_col = "Pays destination"
        elif analyse_type == "Par service":
            groupby_col = "Service de transport"
        else:
            groupby_col = "Nom du partenaire"
        
        analyse_df = commandes_tarifées.groupby(groupby_col).agg({
            "Tarif Total": ["sum", "mean", "count"],
            "Marge": ["sum", "mean"] if "Marge" in commandes_tarifées else None
        }).round(2)
        
        st.dataframe(analyse_df, use_container_width=True)
    
    # Données
    with tabs[2]:
        st.dataframe(commandes_tarifées, use_container_width=True)
    
    # Sans tarifs
    with tabs[3]:
        if not commandes_sans_tarif.empty:
            st.warning(f"{len(commandes_sans_tarif)} commandes sans tarif trouvé")
            st.dataframe(commandes_sans_tarif, use_container_width=True)
        else:
            st.success("Toutes les commandes ont un tarif calculé")
    
    # Export
    with tabs[4]:
        col1, col2 = st.columns(2)
        with col1:
            if not commandes_tarifées.empty:
                st.download_button(
                    "📥 Télécharger commandes avec tarifs",
                    data=convertir_df_en_excel(commandes_tarifées),
                    file_name="commandes_avec_tarifs.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        with col2:
            if not commandes_sans_tarif.empty:
                st.download_button(
                    "📥 Télécharger commandes sans tarif",
                    data=convertir_df_en_excel(commandes_sans_tarif),
                    file_name="commandes_sans_tarifs.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

@st.cache_data
def convertir_df_en_excel(df: pd.DataFrame) -> bytes:
    """Convertit un DataFrame en fichier Excel."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

def main():
    """Fonction principale de l'application."""
    st.title("Application de Calcul des Tarifs de Transport")
    
    # Sidebar pour les paramètres
    with st.sidebar:
        st.subheader("Paramètres de la Taxe Gasoil")
        taxe_par_transporteur = {}
        for transporteur in ["Colissimo", "Chronopost", "DHL", "UPS", "FedEx", "Mondial Relay"]:
            taxe_par_transporteur[transporteur] = st.number_input(
                f"Taxe {transporteur} (%)",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state.get(f'taxe_{transporteur}', 0.0),
                step=0.1,
                key=f'taxe_{transporteur}'
            )
    
    # Zone de téléchargement des fichiers
    with st.expander("📁 Fichiers", expanded=not st.session_state.data_cache['files_uploaded']):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            commandes_file = st.file_uploader("Fichier des commandes", type=['xlsx'])
        with col2:
            tarifs_file = st.file_uploader("Fichier des tarifs", type=['xlsx'])
        with col3:
            prix_achat_file = st.file_uploader("Fichier des prix d'achat (optionnel)", type=['xlsx'])
        
        if commandes_file and tarifs_file:
            if not st.session_state.data_cache['files_uploaded']:
                st.session_state.data_cache = {
                    'commandes': load_data_optimized(commandes_file, DTYPE_CONFIGS['commandes']),
                    'tarifs': load_data_optimized(tarifs_file, DTYPE_CONFIGS['tarifs']),
                    'prix_achat': load_data_optimized(prix_achat_file) if prix_achat_file else None,
                    'files_uploaded': True,
                    'calculation_done': False,
                    'results': None
                }
                st.success("✅ Fichiers chargés avec succès!")

    # Bouton d'exécution des calculs
    if st.session_state.data_cache['files_uploaded'] and not st.session_state.data_cache['calculation_done']:
        if st.button("🚀 Lancer le calcul des tarifs", type="primary"):
            try:
                with st.spinner("Calcul des tarifs en cours..."):
                    # Mesure du temps d'exécution
                    start_time = time.time()
                    
                    # Calcul des tarifs optimisé
                    commandes_tarifées, commandes_sans_tarif = application_calcul_tarif_optimise(
                        st.session_state.data_cache['commandes'],
                        st.session_state.data_cache['tarifs'],
                        taxe_par_transporteur,
                        st.session_state.data_cache['prix_achat']
                    )
                    
                    # Calcul des métriques et graphiques
                    metrics = create_summary_metrics(commandes_tarifées)
                    charts = create_charts(commandes_tarifées)
                    
                    # Stockage des résultats dans le cache
                    st.session_state.data_cache['results'] = {
                        'commandes_tarifées': commandes_tarifées,
                        'commandes_sans_tarif': commandes_sans_tarif,
                        'metrics': metrics,
                        'charts': charts
                    }
                    
                    execution_time = time.time() - start_time
                    st.session_state.data_cache['calculation_done'] = True
                    
                    st.success(f"✨ Calculs terminés en {execution_time:.2f} secondes!")
                    
            except Exception as e:
                st.error(f"❌ Erreur lors du calcul : {str(e)}")
                st.info("📝 Vérifiez le format des fichiers et réessayez.")
    
    # Affichage des résultats
    if st.session_state.data_cache['calculation_done']:
        results = st.session_state.data_cache['results']
        
        # Création des onglets d'analyse
        create_analysis_tabs(
            results['commandes_tarifées'],
            results['commandes_sans_tarif'],
            results['metrics'],
            results['charts']
        )
        
        # Bouton pour réinitialiser les calculs
        if st.button("🔄 Recalculer avec de nouveaux paramètres"):
            st.session_state.data_cache['calculation_done'] = False
            st.rerun()
        
        # Bouton pour réinitialiser complètement
        if st.button("🗑️ Réinitialiser et charger de nouveaux fichiers"):
            st.session_state.data_cache = {
                'commandes': None,
                'tarifs': None,
                'prix_achat': None,
                'results': None,
                'files_uploaded': False,
                'calculation_done': False
            }
            st.rerun()
    
    # Affichage des informations d'aide
    with st.expander("ℹ️ Aide et informations"):
        st.markdown("""
        ### Guide d'utilisation
        1. Téléchargez vos fichiers Excel dans la section 'Fichiers'
        2. Ajustez les taxes gasoil dans le menu latéral si nécessaire
        3. Cliquez sur 'Lancer le calcul des tarifs'
        4. Consultez les résultats dans les différents onglets
        
        ### Format des fichiers requis
        
        **Fichier des commandes** :
        - Nom du partenaire
        - Service de transport
        - Pays destination
        - Poids expédition
        - Code Pays (optionnel)
        
        **Fichier des tarifs** :
        - Partenaire
        - Service
        - Pays
        - PoidsMin
        - PoidsMax
        - Prix
        
        **Fichier des prix d'achat (optionnel)** :
        - Service de transport
        - Prix Achat
        - Code Pays (optionnel)
        """)

if __name__ == "__main__":
    main()
