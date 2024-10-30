import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from typing import Tuple, Dict, Optional, List
import time

# Configuration de la page
st.set_page_config(
    page_title="Calcul des Tarifs de Transport",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration des types de données
DTYPE_CONFIGS = {
    'commandes': {
        'Nom du partenaire': 'string',
        'Service de transport': 'string',
        'Pays destination': 'string',
        'Code Pays': 'string'
    },
    'tarifs': {
        'Partenaire': 'string',
        'Service': 'string',
        'Pays': 'string'
    }
}

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

# Fonction de standardisation des chaînes
@st.cache_data
def standardize_string(s: str) -> str:
    """Standardise les chaînes de caractères pour la comparaison."""
    if pd.isna(s):
        return ''
    return str(s).strip().upper()

@st.cache_data(ttl=3600)
def load_data_optimized(file_uploader, dtype_config=None) -> pd.DataFrame:
    """Charge et optimise les données depuis un fichier."""
    try:
        df = pd.read_excel(
            file_uploader,
            dtype=dtype_config,
            engine='openpyxl'
        )
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def calculer_tarifs_vectorise(commandes: pd.DataFrame, tarifs: pd.DataFrame, taxe_par_transporteur: dict) -> pd.DataFrame:
    """Calcule les tarifs avec une correspondance stricte des partenaires."""
    # Copies de travail
    commandes = commandes.copy()
    tarifs = tarifs.copy()
    
    # Standardisation des colonnes pour la correspondance
    commandes['Nom_partenaire_std'] = commandes['Nom du partenaire'].apply(standardize_string)
    commandes['Service_std'] = commandes['Service de transport'].apply(standardize_string)
    commandes['Pays_std'] = commandes['Pays destination'].apply(standardize_string)
    
    tarifs['Partenaire_std'] = tarifs['Partenaire'].apply(standardize_string)
    tarifs['Service_std'] = tarifs['Service'].apply(standardize_string)
    tarifs['Pays_std'] = tarifs['Pays'].apply(standardize_string)
    
    # Vérification des correspondances
    with st.expander("Vérification des correspondances"):
        st.write("Partenaires dans les commandes:", sorted(commandes['Nom_partenaire_std'].unique()))
        st.write("Partenaires dans les tarifs:", sorted(tarifs['Partenaire_std'].unique()))
    
    # Fusion avec correspondance exacte
    merged = pd.merge(
        commandes,
        tarifs,
        left_on=['Nom_partenaire_std', 'Service_std', 'Pays_std'],
        right_on=['Partenaire_std', 'Service_std', 'Pays_std'],
        how='left',
        validate='many_to_one'
    )
    
    # Vérification des conditions de poids
    poids_condition = (
        (merged['Poids expédition'] >= merged['PoidsMin']) & 
        (merged['Poids expédition'] <= merged['PoidsMax'])
    )
    
    # Affichage des commandes sans correspondance
    no_match = merged[merged['Prix'].isna()]
    if not no_match.empty:
        with st.expander("Commandes sans correspondance"):
            st.dataframe(no_match[['Nom du partenaire', 'Service de transport', 
                                 'Pays destination', 'Poids expédition']].head())
    
    # Application des conditions
    merged.loc[~poids_condition, 'Prix'] = np.nan
    
    # Calcul des taxes
    merged['Taxe'] = 0.0
    for transporteur, taux in taxe_par_transporteur.items():
        mask = merged['Service_std'].str.contains(standardize_string(transporteur), regex=False)
        merged.loc[mask, 'Taxe'] = taux
    
    # Création des résultats
    results = pd.DataFrame(index=commandes.index)
    results['Tarif de Base'] = merged['Prix']
    results['Taxe Gasoil'] = merged.apply(
        lambda row: row['Prix'] * row['Taxe'] / 100 if pd.notnull(row['Prix']) else np.nan,
        axis=1
    )
    results['Tarif Total'] = results['Tarif de Base'] + results['Taxe Gasoil']
    
    return results

@st.cache_data
def application_calcul_tarif_optimise(
    commandes: pd.DataFrame,
    tarifs: pd.DataFrame,
    taxe_par_transporteur: dict,
    prix_achat_df: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calcul optimisé des tarifs avec vérifications."""
    try:
        # Conversion des types numériques
        commandes = commandes.copy()
        tarifs = tarifs.copy()
        
        commandes['Poids expédition'] = pd.to_numeric(commandes['Poids expédition'], errors='coerce')
        tarifs['Prix'] = pd.to_numeric(tarifs['Prix'], errors='coerce')
        tarifs['PoidsMin'] = pd.to_numeric(tarifs['PoidsMin'], errors='coerce')
        tarifs['PoidsMax'] = pd.to_numeric(tarifs['PoidsMax'], errors='coerce')
        
        # Calcul des tarifs
        resultats = calculer_tarifs_vectorise(commandes, tarifs, taxe_par_transporteur)
        commandes[['Tarif de Base', 'Taxe Gasoil', 'Tarif Total']] = resultats
        
        # Séparation des résultats
        mask_tarif_trouve = commandes['Tarif de Base'].notna()
        commandes_tarifées = commandes[mask_tarif_trouve].copy()
        commandes_sans_tarif = commandes[~mask_tarif_trouve].copy()
        
        # Calcul des marges
        if prix_achat_df is not None and not prix_achat_df.empty:
            prix_achat_df = prix_achat_df.copy()
            merge_cols = ['Service de transport']
            
            if 'Code Pays' in prix_achat_df.columns and 'Code Pays' in commandes_tarifées.columns:
                merge_cols.append('Code Pays')
            
            prix_achat_df['Service de transport'] = prix_achat_df['Service de transport'].apply(standardize_string)
            commandes_tarifées['Service_std'] = commandes_tarifées['Service de transport'].apply(standardize_string)
            
            commandes_tarifées = pd.merge(
                commandes_tarifées,
                prix_achat_df,
                left_on=['Service_std'] + (['Code Pays'] if 'Code Pays' in merge_cols else []),
                right_on=['Service de transport'] + (['Code Pays'] if 'Code Pays' in merge_cols else []),
                how='left'
            )
            
            commandes_tarifées['Prix d\'Achat'] = commandes_tarifées['Prix Achat']
            commandes_tarifées['Marge'] = commandes_tarifées['Tarif Total'] - commandes_tarifées['Prix d\'Achat']
            commandes_tarifées['Marge %'] = (commandes_tarifées['Marge'] / commandes_tarifées['Tarif Total'] * 100).round(2)
        
        return commandes_tarifées, commandes_sans_tarif
        
    except Exception as e:
        st.error(f"Erreur lors du calcul des tarifs : {str(e)}")
        st.exception(e)
        raise e

@st.cache_data
def create_summary_metrics(commandes_tarifées: pd.DataFrame) -> Dict:
    """Création des métriques résumées."""
    metrics = {
        "total_commandes": len(commandes_tarifées),
        "ca_total": commandes_tarifées["Tarif Total"].sum(),
    }
    
    if "Marge" in commandes_tarifées.columns:
        metrics.update({
            "marge_totale": commandes_tarifées["Marge"].sum(),
            "marge_moyenne": commandes_tarifées["Marge"].mean(),
            "marge_pourcentage": (commandes_tarifées["Marge"].sum() / commandes_tarifées["Tarif Total"].sum() * 100)
        })
    
    return metrics

@st.cache_data
def create_charts(commandes_tarifées: pd.DataFrame) -> Dict:
    """Création des visualisations."""
    charts = {}
    
    # Graphique des tarifs par partenaire
    partenaire_data = commandes_tarifées.groupby("Nom du partenaire")["Tarif Total"].sum().reset_index()
    charts["tarifs_partenaire"] = px.bar(
        partenaire_data,
        x="Nom du partenaire",
        y="Tarif Total",
        title="Tarif total par partenaire",
        labels={"Tarif Total": "Montant (€)", "Nom du partenaire": "Partenaire"}
    )
    
    if "Marge" in commandes_tarifées.columns:
        charts["marges"] = px.scatter(
            commandes_tarifées,
            x="Tarif Total",
            y="Marge",
            color="Service de transport",
            title="Marge vs Tarif",
            labels={"Tarif Total": "Tarif (€)", "Marge": "Marge (€)"}
        )
        
        charts["marges_service"] = px.box(
            commandes_tarifées,
            x="Service de transport",
            y="Marge %",
            title="Distribution des marges par service",
            labels={"Marge %": "Marge (%)", "Service de transport": "Service"}
        )
    
    return charts

@st.cache_data
def convertir_df_en_excel(df: pd.DataFrame) -> bytes:
    """Conversion d'un DataFrame en fichier Excel."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()
 def create_analysis_tabs(commandes_tarifées, commandes_sans_tarif, metrics, charts):
    """Création des onglets d'analyse."""
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
            st.metric("Total commandes", f"{metrics['total_commandes']:,}")
            st.metric("CA Total", f"{metrics['ca_total']:,.2f} €")
        with col2:
            if "marge_totale" in metrics:
                st.metric("Marge totale", f"{metrics['marge_totale']:,.2f} €")
                st.metric("Marge moyenne", f"{metrics['marge_moyenne']:,.2f} €")
        with col3:
            if "marge_pourcentage" in metrics:
                st.metric("Marge globale", f"{metrics['marge_pourcentage']:.2f}%")
            st.metric("Commandes sans tarif", len(commandes_sans_tarif))
        
        st.plotly_chart(charts["tarifs_partenaire"], use_container_width=True)
        if "marges" in charts:
            st.plotly_chart(charts["marges"], use_container_width=True)
    
    # Analyses détaillées
    with tabs[1]:
        if "marges_service" in charts:
            st.plotly_chart(charts["marges_service"], use_container_width=True)
        
        analyse_type = st.selectbox(
            "Type d'analyse",
            ["Par pays", "Par service", "Par partenaire"]
        )
        
        agg_cols = {
            "Tarif Total": ["sum", "mean", "count"]
        }
        if "Marge" in commandes_tarifées.columns:
            agg_cols["Marge"] = ["sum", "mean"]
        
        group_cols = {
            "Par pays": "Pays destination",
            "Par service": "Service de transport",
            "Par partenaire": "Nom du partenaire"
        }
        groupby_col = group_cols[analyse_type]
        
        try:
            analyse_df = commandes_tarifées.groupby(groupby_col).agg(agg_cols)
            new_cols = []
            for col, aggs in agg_cols.items():
                for agg in aggs:
                    new_cols.append(f"{agg.capitalize()} {col}")
            
            analyse_df.columns = analyse_df.columns.droplevel()
            analyse_df.columns = new_cols
            analyse_df = analyse_df.round(2)
            
            st.dataframe(analyse_df, use_container_width=True)
            
            fig = px.bar(
                analyse_df.reset_index(),
                x=groupby_col,
                y=new_cols[0],
                title=f"Analyse {analyse_type}",
                labels={new_cols[0]: "Montant (€)"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Erreur lors de l'analyse: {str(e)}")
    
    # Données
    with tabs[2]:
        display_cols = [
            "Nom du partenaire",
            "Service de transport",
            "Pays destination",
            "Poids expédition",
            "Tarif de Base",
            "Taxe Gasoil",
            "Tarif Total"
        ]
        
        if "Code Pays" in commandes_tarifées.columns:
            display_cols.insert(3, "Code Pays")
        if "Marge" in commandes_tarifées.columns:
            display_cols.extend(["Prix d'Achat", "Marge", "Marge %"])
        
        st.dataframe(commandes_tarifées[display_cols], use_container_width=True)
    
    # Sans tarifs
    with tabs[3]:
        if not commandes_sans_tarif.empty:
            st.warning(f"{len(commandes_sans_tarif)} commandes sans tarif trouvé")
            display_cols_sans_tarif = [
                "Nom du partenaire",
                "Service de transport",
                "Pays destination",
                "Poids expédition"
            ]
            if "Code Pays" in commandes_sans_tarif.columns:
                display_cols_sans_tarif.insert(3, "Code Pays")
            
            st.dataframe(commandes_sans_tarif[display_cols_sans_tarif], use_container_width=True)
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
            commandes_file = st.file_uploader(
                "Fichier des commandes",
                type=['xlsx'],
                help="Fichier Excel contenant les commandes à traiter"
            )
        
        with col2:
            tarifs_file = st.file_uploader(
                "Fichier des tarifs",
                type=['xlsx'],
                help="Fichier Excel contenant les grilles tarifaires"
            )
        
        with col3:
            prix_achat_file = st.file_uploader(
                "Fichier des prix d'achat (optionnel)",
                type=['xlsx'],
                help="Fichier Excel contenant les prix d'achat pour le calcul des marges"
            )

        if commandes_file and tarifs_file:
            if not st.session_state.data_cache['files_uploaded']:
                with st.spinner("Chargement des fichiers..."):
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
                        st.error(f"Erreur lors du chargement des fichiers: {str(e)}")
                        return

    # Bouton d'exécution des calculs
    if st.session_state.data_cache['files_uploaded'] and not st.session_state.data_cache['calculation_done']:
        if st.button("🚀 Lancer le calcul des tarifs", type="primary", use_container_width=True):
            try:
                with st.spinner("Calcul des tarifs en cours..."):
                    start_time = time.time()
                    
                    # Vérification des colonnes requises
                    required_cols_commandes = ["Nom du partenaire", "Service de transport", "Pays destination", "Poids expédition"]
                    required_cols_tarifs = ["Partenaire", "Service", "Pays", "PoidsMin", "PoidsMax", "Prix"]
                    
                    commandes = st.session_state.data_cache['commandes']
                    tarifs = st.session_state.data_cache['tarifs']
                    
                    missing_cols_commandes = [col for col in required_cols_commandes if col not in commandes.columns]
                    missing_cols_tarifs = [col for col in required_cols_tarifs if col not in tarifs.columns]
                    
                    if missing_cols_commandes or missing_cols_tarifs:
                        if missing_cols_commandes:
                            st.error(f"Colonnes manquantes dans le fichier commandes: {', '.join(missing_cols_commandes)}")
                        if missing_cols_tarifs:
                            st.error(f"Colonnes manquantes dans le fichier tarifs: {', '.join(missing_cols_tarifs)}")
                        return
                    
                    # Calcul des tarifs
                    commandes_tarifées, commandes_sans_tarif = application_calcul_tarif_optimise(
                        commandes,
                        tarifs,
                        taxe_par_transporteur,
                        st.session_state.data_cache['prix_achat']
                    )
                    
                    # Création des métriques et graphiques
                    metrics = create_summary_metrics(commandes_tarifées)
                    charts = create_charts(commandes_tarifées)
                    
                    # Stockage des résultats
                    st.session_state.data_cache['results'] = {
                        'commandes_tarifées': commandes_tarifées,
                        'commandes_sans_tarif': commandes_sans_tarif,
                        'metrics': metrics,
                        'charts': charts
                    }
                    
                    execution_time = time.time() - start_time
                    st.session_state.data_cache['calculation_done'] = True
                    
                    st.success(f"✨ Calculs terminés en {execution_time:.2f} secondes!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Erreur lors du calcul : {str(e)}")
                st.info("📝 Vérifiez le format des fichiers et réessayez.")

    # Affichage des résultats
    if st.session_state.data_cache['calculation_done']:
        results = st.session_state.data_cache['results']
        
        create_analysis_tabs(
            results['commandes_tarifées'],
            results['commandes_sans_tarif'],
            results['metrics'],
            results['charts']
        )
        
        # Boutons de réinitialisation
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Recalculer", use_container_width=True):
                st.session_state.data_cache['calculation_done'] = False
                st.rerun()
        
        with col2:
            if st.button("🗑️ Réinitialiser", use_container_width=True):
                st.session_state.data_cache = {
                    'commandes': None,
                    'tarifs': None,
                    'prix_achat': None,
                    'results': None,
                    'files_uploaded': False,
                    'calculation_done': False
                }
                st.rerun()

    # Aide et informations
    with st.expander("ℹ️ Aide et informations"):
        st.markdown("""
        ### Guide d'utilisation
        1. Téléchargez vos fichiers Excel dans la section 'Fichiers'
        2. Ajustez les taxes gasoil dans le menu latéral si nécessaire
        3. Cliquez sur 'Lancer le calcul des tarifs'
        4. Consultez les résultats dans les différents onglets
        
        ### Format des fichiers requis
        
        **Fichier des commandes**:
        - Nom du partenaire
        - Service de transport
        - Pays destination
        - Poids expédition
        - Code Pays (optionnel)
        
        **Fichier des tarifs**:
        - Partenaire
        - Service
        - Pays
        - PoidsMin
        - PoidsMax
        - Prix
        
        **Fichier des prix d'achat (optionnel)**:
        - Service de transport
        - Prix Achat
        - Code Pays (optionnel)
        """)

if __name__ == "__main__":
    main()
