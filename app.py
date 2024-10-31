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

@st.cache_data
def load_data(file_uploader, usecols: Optional[list] = None) -> pd.DataFrame:
    """Charge et met en cache les données avec les colonnes spécifiées."""
    return pd.read_excel(file_uploader, usecols=usecols)

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
    taxe = taxe_par_transporteur.get(service.lower(), 0)
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
    results = []
    for start_idx in range(0, len(commandes), batch_size):
        batch = commandes.iloc[start_idx:start_idx + batch_size]
        batch_results = [
            calculer_tarif_avec_taxe(
                row["Nom du partenaire"],
                row["Service de transport"],
                row["Pays destination"],
                row["Poids expédition"],
                tarifs,
                taxe_par_transporteur
            )
            for _, row in batch.iterrows()
        ]
        results.extend(batch_results)
        st.progress(min((start_idx + batch_size) / len(commandes), 1.0))

    commandes_processed = commandes.copy()
    commandes_processed[["Tarif de Base", "Taxe Gasoil", "Tarif Total"]] = pd.DataFrame(results)
    mask_tarif_trouve = commandes_processed["Tarif de Base"] != "Tarif non trouvé"
    commandes_tarifées = commandes_processed[mask_tarif_trouve].copy()
    commandes_sans_tarif = commandes_processed[~mask_tarif_trouve].copy()
    commandes_tarifées[["Tarif de Base", "Taxe Gasoil", "Tarif Total"]] = commandes_tarifées[
        ["Tarif de Base", "Taxe Gasoil", "Tarif Total"]
    ].apply(pd.to_numeric, errors="coerce")
    
    # Vérification des colonnes dans le fichier de prix d'achat
    if prix_achat_df is not None:
        required_columns = ["Service", "Pays", "PoidsMin", "PoidsMax", "Prix d'Achat"]
        missing_columns = [col for col in required_columns if col not in prix_achat_df.columns]
        
        if missing_columns:
            st.error(f"Colonnes manquantes dans le fichier de prix d'achat: {', '.join(missing_columns)}")
            return commandes_tarifées, commandes_sans_tarif
        
        # Merge les données en utilisant la tranche de poids appropriée
        commandes_tarifées = commandes_tarifées.merge(
            prix_achat_df,
            left_on=["Service de transport", "Pays destination"],
            right_on=["Service", "Pays"],
            how="left"
        )
        
        # Filtre pour associer chaque commande à sa tranche de poids correcte
        commandes_tarifées = commandes_tarifées[
            (commandes_tarifées["Poids expédition"] >= commandes_tarifées["PoidsMin"]) &
            (commandes_tarifées["Poids expédition"] <= commandes_tarifées["PoidsMax"])
        ]
        
        # Calcul de la marge
        commandes_tarifées["Marge"] = commandes_tarifées["Tarif Total"] - commandes_tarifées["Prix d'Achat"]
    else:
        commandes_tarifées["Prix d'Achat"] = np.nan
        commandes_tarifées["Marge"] = np.nan

    return commandes_tarifées, commandes_sans_tarif

@st.cache_data
def load_data(file_uploader, usecols: Optional[list] = None) -> pd.DataFrame:
    """Charge et met en cache les données depuis un fichier avec vérification des colonnes."""
    data = pd.read_excel(file_uploader)
    # Affiche les colonnes disponibles pour aider au débogage
    st.write("Colonnes disponibles dans le fichier :", data.columns.tolist())
    if usecols:
        # Vérifie que toutes les colonnes demandées sont présentes dans le fichier
        missing_cols = [col for col in usecols if col not in data.columns]
        if missing_cols:
            st.error(f"Colonnes manquantes dans le fichier : {', '.join(missing_cols)}")
            raise ValueError(f"Colonnes manquantes dans le fichier : {', '.join(missing_cols)}")
        data = pd.read_excel(file_uploader, usecols=usecols)
    return data

@st.cache_data
def process_weight_control(compiled_file, facture_file) -> pd.DataFrame:
    """Analyse les écarts de poids entre les données compilées et facturées avec gestion des colonnes manquantes."""
    compiled_data = load_data(compiled_file, usecols=["Tracking", "Poids expédition"])
    facture_data = load_data(facture_file, usecols=["Tracking", "Poids_facture"])

    # Renommage des colonnes si nécessaire pour garantir la compatibilité
    compiled_data = compiled_data.rename(columns={"Poids expédition": "Poids_expédition"})
    facture_data = facture_data.rename(columns={"Poids constaté": "Poids_facture"})

    # Fusion et calcul de l'écart de poids
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

def display_results_tab(commandes_tarifées: pd.DataFrame, commandes_sans_tarif: pd.DataFrame):
    st.subheader("Commandes avec Tarifs Calculés")
    st.dataframe(commandes_tarifées, height=400, use_container_width=True)

    if not commandes_sans_tarif.empty:
        st.subheader("Commandes Sans Tarifs")
        st.dataframe(commandes_sans_tarif, height=200, use_container_width=True)
        st.warning(f"Nombre de commandes sans tarif trouvé : {len(commandes_sans_tarif)}")
    else:
        st.success("Toutes les commandes ont un tarif calculé")

def display_graphics_tab(commandes_tarifées: pd.DataFrame):
    st.subheader("Analyses Graphiques")
    chart_type = st.selectbox("Sélectionner le type d'analyse", ["Coût par Partenaire", "Coût par Pays", "Coût par Service"])
    if chart_type == "Coût par Partenaire":
        data = commandes_tarifées.groupby("Nom du partenaire")["Tarif Total"].sum().reset_index()
        fig = px.bar(data, x="Nom du partenaire", y="Tarif Total", title="Coût total par Partenaire")
    elif chart_type == "Coût par Pays":
        data = commandes_tarifées.groupby("Pays destination")["Tarif Total"].sum().reset_index()
        fig = px.bar(data, x="Pays destination", y="Tarif Total", title="Coût total par Pays")
    else:
        data = commandes_tarifées.groupby("Service de transport")["Tarif Total"].sum().reset_index()
        fig = px.bar(data, x="Service de transport", y="Tarif Total", title="Coût total par Service")
    st.plotly_chart(fig, use_container_width=True)

def display_analysis_tab(commandes_tarifées: pd.DataFrame):
    st.subheader("Analyse de Marge")
    st.metric("Marge Totale", f"{commandes_tarifées['Marge'].sum():.2f} €")
    st.metric("Marge Moyenne", f"{commandes_tarifées['Marge'].mean():.2f} €")
    st.metric("Nombre de Commandes", len(commandes_tarifées))

def display_weight_control_tab(compiled_file, facture_file):
    """Affiche l'analyse des écarts de poids."""
    merged_data = process_weight_control(compiled_file, facture_file)
    
    st.subheader("Analyse des Écarts de Poids")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Écart Moyen", f"{merged_data['Ecart_Poids'].mean():.2f} kg")
    with col2:
        st.metric("Écart Total", f"{merged_data['Ecart_Poids'].sum():.2f} kg")
    with col3:
        st.metric("Nombre d'Écarts", len(merged_data[merged_data['Ecart_Poids'] != 0]))

def display_download_tab(commandes_tarifées: pd.DataFrame, commandes_sans_tarif: pd.DataFrame):
    st.subheader("Téléchargement des Résultats")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Télécharger commandes avec tarifs", convertir_df_en_excel(commandes_tarifées), "commandes_avec_tarifs.xlsx")
    with col2:
        st.download_button("📥 Télécharger commandes sans tarif", convertir_df_en_excel(commandes_sans_tarif), "commandes_sans_tarifs.xlsx")

def main():
    """Fonction principale de l'application."""
    st.title("Application de Calcul des Tarifs de Transport")
    taxe_par_transporteur = {
        transporteur: st.sidebar.number_input(f"Taxe {transporteur} (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
        for transporteur in ["Colissimo", "Chronopost", "DHL", "UPS", "FedEx", "Mondial Relay"]
    }
    files = {
        'commandes': st.file_uploader("Fichier de commandes", type=["xlsx"]),
        'tarifs': st.file_uploader("Fichier de tarifs", type=["xlsx"]),
        'prix_achat': st.file_uploader("Fichier prix d'achat (optionnel)", type=["xlsx"]),
        'compiled': st.file_uploader("Fichier compilé", type=["xlsx"]),
        'facture': st.file_uploader("Fichier de facturation", type=["xlsx"])
    }

    if files['commandes'] and files['tarifs']:
        commandes = load_data(files['commandes'], usecols=["Nom du partenaire", "Service de transport", "Pays destination", "Poids expédition"])
        tarifs = load_data(files['tarifs'], usecols=["Partenaire", "Service", "Pays", "PoidsMin", "PoidsMax", "Prix"])
        prix_achat_df = load_data(files['prix_achat']) if files['prix_achat'] else None
        commandes_tarifées, commandes_sans_tarif = application_calcul_tarif_batch(commandes, tarifs, taxe_par_transporteur, prix_achat_df)

        tabs = st.tabs(["📊 Résultats", "📈 Graphiques", "💾 Téléchargement", "💰 Analyse", "⚖️ Contrôle"])
        with tabs[0]:
            display_results_tab(commandes_tarifées, commandes_sans_tarif)
        with tabs[1]:
            display_graphics_tab(commandes_tarifées)
        with tabs[2]:
            display_download_tab(commandes_tarifées, commandes_sans_tarif)
        with tabs[3]:
            display_analysis_tab(commandes_tarifées)
        with tabs[4]:
            if files['compiled'] and files['facture']:
                display_weight_control_tab(files['compiled'], files['facture'])
            else:
                st.info("Veuillez télécharger les fichiers compilé et de facturation pour le contrôle de poids.")

if __name__ == "__main__":
    main()
