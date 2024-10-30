import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from typing import Tuple, Dict, Optional

# Configuration de la page
st.set_page_config(
    page_title="Calcul des Tarifs de Transport",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fonctions utilitaires avec cache
@st.cache_data
def load_data(file_uploader) -> pd.DataFrame:
    """Charge et met en cache les données depuis un fichier uploadé."""
    return pd.read_excel(file_uploader)

@st.cache_data
def calculer_tarif_avec_taxe(
    partenaire: str,
    service: str, 
    pays: str,
    poids: float,
    tarifs: pd.DataFrame,
    taxe_par_transporteur: Dict[str, float]
) -> Tuple[float, float, float]:
    """Calcule le tarif avec taxe pour une expédition."""
    tarif_ligne = tarifs[
        (tarifs["Partenaire"] == partenaire) &
        (tarifs["Service"] == service) &
        (tarifs["Pays"] == pays) &
        (tarifs["PoidsMin"] <= poids) &
        (tarifs["PoidsMax"] >= poids)
    ]
    
    if tarif_ligne.empty:
        return "Tarif non trouvé", "Tarif non trouvé", "Tarif non trouvé"
        
    tarif_base = tarif_ligne.iloc[0]["Prix"]
    taxe = next(
        (taux for transporteur, taux in taxe_par_transporteur.items() 
         if transporteur.lower() in service.lower()),
        0
    )
    taxe_gasoil = tarif_base * (taxe / 100)
    return tarif_base, taxe_gasoil, tarif_base + taxe_gasoil

@st.cache_data
def application_calcul_tarif_batch(
    commandes: pd.DataFrame,
    tarifs: pd.DataFrame,
    taxe_par_transporteur: Dict[str, float],
    prix_achat_df: Optional[pd.DataFrame] = None,
    batch_size: int = 1000
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Traite les commandes par lots pour de meilleures performances."""
    has_code_pays = "Code Pays" in commandes.columns
    results = []
    
    # Traitement par lots
    for start_idx in range(0, len(commandes), batch_size):
        batch = commandes.iloc[start_idx:start_idx + batch_size]
        batch_results = []
        
        for _, row in batch.iterrows():
            tarifs_batch = calculer_tarif_avec_taxe(
                row["Nom du partenaire"],
                row["Service de transport"],
                row["Pays destination"],
                row["Poids expédition"],
                tarifs,
                taxe_par_transporteur
            )
            batch_results.append(tarifs_batch)
        
        results.extend(batch_results)
        
        # Indication de progression
        progress = min((start_idx + batch_size) / len(commandes), 1.0)
        st.progress(progress)
    
    commandes_processed = commandes.copy()
    commandes_processed[["Tarif de Base", "Taxe Gasoil", "Tarif Total"]] = pd.DataFrame(results)
    
    # Séparation et conversion
    mask_tarif_trouve = commandes_processed["Tarif de Base"] != "Tarif non trouvé"
    commandes_tarifées = commandes_processed[mask_tarif_trouve].copy()
    commandes_sans_tarif = commandes_processed[~mask_tarif_trouve].copy()
    
    # Conversion numérique
    for col in ["Tarif de Base", "Taxe Gasoil", "Tarif Total"]:
        commandes_tarifées[col] = pd.to_numeric(commandes_tarifées[col], errors='coerce')
    
    # Calcul des marges
    if prix_achat_df is not None:
        merge_cols = ["Service de transport"]
        if has_code_pays and "Code Pays" in prix_achat_df.columns:
            merge_cols.append("Code Pays")
            
        commandes_tarifées = commandes_tarifées.merge(
            prix_achat_df,
            on=merge_cols,
            how="left"
        ).rename(columns={"Prix Achat": "Prix d'Achat"})
        
        commandes_tarifées["Marge"] = commandes_tarifées["Tarif Total"] - commandes_tarifées["Prix d'Achat"]
    else:
        commandes_tarifées["Prix d'Achat"] = np.nan
        commandes_tarifées["Marge"] = np.nan
    
    return commandes_tarifées, commandes_sans_tarif

@st.cache_data
def process_weight_control(
    compiled_file,
    facture_file
) -> pd.DataFrame:
    """Traite et met en cache les données de contrôle de poids."""
    compiled_data = load_data(compiled_file)
    facture_data = load_data(facture_file)
    
    # Renommage des colonnes
    compiled_data = compiled_data.rename(columns={
        "Numéro de commande partenaire": "Tracking",
        "Poids expédition": "Poids_expédition"
    })
    facture_data = facture_data.rename(columns={
        "Poids constaté": "Poids_facture"
    })
    
    # Conversion des types
    compiled_data["Tracking"] = compiled_data["Tracking"].astype(str)
    facture_data["Tracking"] = facture_data["Tracking"].astype(str)
    
    # Fusion et calcul
    merged_data = pd.merge(facture_data, compiled_data, on="Tracking", how="left")
    merged_data["Ecart_Poids"] = merged_data["Poids_facture"] - merged_data["Poids_expédition"]
    
    return merged_data

@st.cache_data
def convertir_df_en_excel(df: pd.DataFrame, include_marge: bool = False) -> bytes:
    """Convertit un DataFrame en fichier Excel."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        if not include_marge:
            df = df.drop(columns=["Prix d'Achat", "Marge"], errors="ignore")
        df.to_excel(writer, index=False)
    return output.getvalue()

# Fonctions d'affichage
def display_results_tab(
    commandes_tarifées: pd.DataFrame,
    commandes_sans_tarif: pd.DataFrame
):
    """Affiche l'onglet des résultats avec filtres."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Filtres pour les commandes avec tarifs")
        filters = {
            "Partenaire": st.multiselect(
                "Partenaire",
                options=sorted(commandes_tarifées["Nom du partenaire"].unique())
            ),
            "Service": st.multiselect(
                "Service de transport",
                options=sorted(commandes_tarifées["Service de transport"].unique())
            ),
            "Pays": st.multiselect(
                "Pays destination",
                options=sorted(commandes_tarifées["Pays destination"].unique())
            )
        }
    
    with col2:
        st.subheader("Statistiques")
        st.metric("Total commandes", len(commandes_tarifées) + len(commandes_sans_tarif))
        st.metric("Commandes sans tarif", len(commandes_sans_tarif))
        if len(commandes_tarifées) > 0:
            st.metric("Montant total", f"{commandes_tarifées['Tarif Total'].sum():.2f} €")
    
    # Application des filtres
    filtered_data = commandes_tarifées.copy()
    for col, selected in filters.items():
        if selected:
            col_name = "Nom du partenaire" if col == "Partenaire" else \
                      "Service de transport" if col == "Service" else \
                      "Pays destination"
            filtered_data = filtered_data[filtered_data[col_name].isin(selected)]
    
    # Affichage des résultats
    st.subheader("Commandes avec Tarifs Calculés")
    display_columns = ["Nom du partenaire", "Service de transport", 
                      "Pays destination", "Poids expédition",
                      "Tarif de Base", "Taxe Gasoil", "Tarif Total"]
    
    if "Code Pays" in filtered_data.columns:
        display_columns.insert(3, "Code Pays")
    
    st.dataframe(
        filtered_data[display_columns],
        height=400,
        use_container_width=True
    )

    if not commandes_sans_tarif.empty:
        st.subheader("Commandes Sans Tarifs")
        display_columns_sans_tarif = ["Nom du partenaire", "Service de transport", 
                                    "Pays destination", "Poids expédition"]
        if "Code Pays" in commandes_sans_tarif.columns:
            display_columns_sans_tarif.insert(3, "Code Pays")
        
        st.dataframe(
            commandes_sans_tarif[display_columns_sans_tarif],
            height=200,
            use_container_width=True
        )
        st.warning(f"Nombre de commandes sans tarif trouvé : {len(commandes_sans_tarif)}")
    else:
        st.success("Toutes les commandes ont un tarif calculé")

def display_graphics_tab(commandes_tarifées: pd.DataFrame):
    """Affiche les graphiques d'analyse."""
    st.subheader("Analyses Graphiques")
    
    chart_type = st.selectbox(
        "Sélectionner le type d'analyse",
        ["Coût par Partenaire", "Coût par Pays", "Coût par Service"]
    )
    
    if not commandes_tarifées.empty:
        if chart_type == "Coût par Partenaire":
            data = commandes_tarifées.groupby("Nom du partenaire")["Tarif Total"].agg(
                ['sum', 'mean', 'count']
            ).reset_index()
            fig = px.bar(
                data,
                x="Nom du partenaire",
                y="sum",
                title="Coût total par Partenaire",
                labels={
                    "sum": "Coût total (€)",
                    "Nom du partenaire": "Partenaire"
                }
            )
        elif chart_type == "Coût par Pays":
            data = commandes_tarifées.groupby("Pays destination")["Tarif Total"].agg(
                ['sum', 'mean', 'count']
            ).reset_index()
            fig = px.bar(
                data,
                x="Pays destination",
                y="sum",
                title="Coût total par Pays",
                labels={
                    "sum": "Coût total (€)",
                    "Pays destination": "Pays"
                }
            )
        else:
            data = commandes_tarifées.groupby("Service de transport")["Tarif Total"].agg(
                ['sum', 'mean', 'count']
            ).reset_index()
            fig = px.bar(
                data,
                x="Service de transport",
                y="sum",
                title="Coût total par Service",
                labels={
                    "sum": "Coût total (€)",
                    "Service de transport": "Service"
                }
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Affichage des statistiques détaillées
        st.subheader("Statistiques détaillées")
        st.dataframe(
            data.rename(columns={
                'sum': 'Total',
                'mean': 'Moyenne',
                'count': 'Nombre'
            }).round(2),
            use_container_width=True
        )

def display_download_tab(
    commandes_tarifées: pd.DataFrame,
    commandes_sans_tarif: pd.DataFrame
):
    """Affiche les options de téléchargement."""
    st.subheader("Téléchargement des Résultats")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            "📥 Télécharger commandes avec tarifs",
            convertir_df_en_excel(commandes_tarifées),
            "commandes_avec_tarifs.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col2:
        st.download_button(
            "📥 Télécharger commandes sans tarif",
            convertir_df_en_excel(commandes_sans_tarif),
            "commandes_sans_tarifs.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

def display_analysis_tab(commandes_tarifées: pd.DataFrame):
    """Affiche l'analyse des marges."""
    st.subheader("Analyse de Marge")
    
    if "Code Pays" in commandes_tarifées.columns:
        st.subheader("Analyse par Code Pays")
        marge_pays = commandes_tarifées.groupby("Code Pays").agg({
            "Marge": ["sum", "mean", "count"],
            "Tarif Total": "sum",
            "Prix d'Achat": "sum"
        }).round(2)
        
        marge_pays.columns = ["Marge Totale", "Marge Moyenne", "Nombre Commandes", 
                            "CA Total", "Coût Total"]
        marge_pays = marge_pays.reset_index()
        
        st.dataframe(marge_pays, use_container_width=True)
        
        fig = px.bar(
            marge_pays,
            x="Code Pays",
            y="Marge Totale",
            title="Marge Totale par Pays",
            labels={"Marge Totale": "Marge (€)", "Code Pays": "Pays"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Détail des Marges")
    display_columns = [
        "Nom du partenaire", "Service de transport",
        "Pays destination", "Poids expédition",
        "Tarif Total", "Prix d'Achat", "Marge"
    ]
    if "Code Pays" in commandes_tarifées.columns:
        display_columns.insert(3, "
                               display_columns.insert(3, "Code Pays")
        
    st.dataframe(
        commandes_tarifées[display_columns],
        height=400,
        use_container_width=True
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Marge Totale",
            f"{commandes_tarifées['Marge'].sum():.2f} €"
        )
    with col2:
        st.metric(
            "Marge Moyenne",
            f"{commandes_tarifées['Marge'].mean():.2f} €"
        )
    with col3:
        st.metric(
            "Nombre de Commandes",
            len(commandes_tarifées)
        )

def display_weight_control_tab(compiled_file, facture_file):
    """Affiche l'analyse du contrôle de poids."""
    merged_data = process_weight_control(compiled_file, facture_file)
    
    st.subheader("Analyse des Écarts de Poids")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Écart Moyen",
            f"{merged_data['Ecart_Poids'].mean():.2f} kg"
        )
    with col2:
        st.metric(
            "Écart Total",
            f"{merged_data['Ecart_Poids'].sum():.2f} kg"
        )
    with col3:
        st.metric(
            "Nombre d'Écarts",
            len(merged_data[merged_data['Ecart_Poids'] != 0])
        )
    
    # Graphique des écarts
    fig = px.histogram(
        merged_data,
        x="Ecart_Poids",
        title="Distribution des Écarts de Poids",
        labels={"Ecart_Poids": "Écart de Poids (kg)"}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.download_button(
        "📥 Télécharger rapport de contrôle",
        convertir_df_en_excel(merged_data),
        "controle_poids_enrichi.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    st.subheader("Détail des Écarts")
    st.dataframe(merged_data, height=400, use_container_width=True)

def main():
    """Fonction principale de l'application."""
    st.title("Application de Calcul des Tarifs de Transport")
    
    # Sidebar configuration
    st.sidebar.subheader("Paramètres de la Taxe Gasoil")
    mots_cles_transporteurs = ["Colissimo", "Chronopost", "DHL", "UPS", "FedEx", "Mondial Relay"]
    taxe_par_transporteur = {
        transporteur: st.sidebar.number_input(
            f"Taxe {transporteur} (%)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.1,
            key=f"taxe_{transporteur}"
        )
        for transporteur in mots_cles_transporteurs
    }
    
    # État de session pour le cache des fichiers
    if 'files' not in st.session_state:
        st.session_state.files = {
            'commandes': None,
            'tarifs': None,
            'prix_achat': None,
            'compiled': None,
            'facture': None
        }
    
    # File uploaders avec mise en cache
    with st.expander("📁 Téléchargement des Fichiers", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            files = {
                'commandes': st.file_uploader("Fichier de commandes", type=["xlsx"]),
                'tarifs': st.file_uploader("Fichier de tarifs", type=["xlsx"]),
                'prix_achat': st.file_uploader("Fichier prix d'achat (optionnel)", type=["xlsx"])
            }
        with col2:
            files.update({
                'compiled': st.file_uploader("Fichier compilé", type=["xlsx"]),
                'facture': st.file_uploader("Fichier de facturation", type=["xlsx"])
            })
    
    # Mise à jour du cache des fichiers
    for key, file in files.items():
        if file is not None:
            st.session_state.files[key] = file
    
    if st.session_state.files['commandes'] and st.session_state.files['tarifs']:
        try:
            # Chargement et traitement des données avec barre de progression
            with st.spinner("Chargement des données..."):
                commandes = load_data(st.session_state.files['commandes'])
                tarifs = load_data(st.session_state.files['tarifs'])
                prix_achat_df = None
                
                if st.session_state.files['prix_achat']:
                    prix_achat_df = load_data(st.session_state.files['prix_achat'])
            
            # Vérification des colonnes requises
            required_columns_commandes = [
                "Nom du partenaire", "Service de transport",
                "Pays destination", "Poids expédition"
            ]
            required_columns_tarifs = [
                "Partenaire", "Service", "Pays",
                "PoidsMin", "PoidsMax", "Prix"
            ]
            
            missing_cols_commandes = [
                col for col in required_columns_commandes 
                if col not in commandes.columns
            ]
            missing_cols_tarifs = [
                col for col in required_columns_tarifs 
                if col not in tarifs.columns
            ]
            
            if missing_cols_commandes or missing_cols_tarifs:
                if missing_cols_commandes:
                    st.error(f"Colonnes manquantes dans le fichier de commandes: {', '.join(missing_cols_commandes)}")
                if missing_cols_tarifs:
                    st.error(f"Colonnes manquantes dans le fichier de tarifs: {', '.join(missing_cols_tarifs)}")
                return
            
            # Calcul des tarifs avec barre de progression
            with st.spinner("Calcul des tarifs en cours..."):
                commandes_tarifées, commandes_sans_tarif = application_calcul_tarif_batch(
                    commandes, tarifs, taxe_par_transporteur, prix_achat_df
                )
            
            # Création des onglets
            tabs = st.tabs([
                "📊 Résultats",
                "📈 Graphiques",
                "💾 Téléchargement",
                "💰 Analyse",
                "⚖️ Contrôle"
            ])
            
            # Affichage des différents onglets
            with tabs[0]:
                display_results_tab(commandes_tarifées, commandes_sans_tarif)
            
            with tabs[1]:
                display_graphics_tab(commandes_tarifées)
            
            with tabs[2]:
                display_download_tab(commandes_tarifées, commandes_sans_tarif)
            
            with tabs[3]:
                if prix_achat_df is not None:
                    display_analysis_tab(commandes_tarifées)
                else:
                    st.info("Veuillez télécharger le fichier de prix d'achat pour voir l'analyse des marges.")
            
            with tabs[4]:
                if st.session_state.files['compiled'] and st.session_state.files['facture']:
                    display_weight_control_tab(
                        st.session_state.files['compiled'],
                        st.session_state.files['facture']
                    )
                else:
                    st.info("Veuillez télécharger les fichiers compilé et de facturation pour le contrôle de poids.")
                    
        except Exception as e:
            st.error(f"Une erreur s'est produite lors du traitement des données: {str(e)}")
            st.info("Veuillez vérifier le format des fichiers et réessayer.")
            
    else:
        st.info("Veuillez télécharger les fichiers de commandes et de tarifs pour commencer l'analyse.")

if __name__ == "__main__":
    main()
