import streamlit as st
import dask.dataframe as dd
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from typing import Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Calcul des Tarifs de Transport",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fonctions de chargement et de traitement des données volumineuses
@st.cache_data
def load_data(file_uploader, file_format="parquet") -> dd.DataFrame:
    """Charge les données avec Dask pour gérer des ensembles volumineux."""
    if file_format == "parquet":
        return dd.read_parquet(file_uploader)
    elif file_format == "csv":
        return dd.read_csv(file_uploader)
    else:
        raise ValueError("Format de fichier non supporté. Utilisez 'parquet' ou 'csv'.")

@st.cache_data
def calculer_tarif_avec_taxe(
    partenaire: str,
    service: str, 
    pays: str,
    poids: float,
    tarifs: dd.DataFrame,
    taxe_par_transporteur: Dict[str, float]
) -> Tuple[float, float, float]:
    """Calcule le tarif avec taxe pour une expédition spécifique en utilisant Dask."""
    tarif_ligne = tarifs[
        (tarifs["Partenaire"] == partenaire) &
        (tarifs["Service"] == service) &
        (tarifs["Pays"] == pays) &
        (tarifs["PoidsMin"] <= poids) &
        (tarifs["PoidsMax"] >= poids)
    ].compute()
    
    if tarif_ligne.empty:
        return "Tarif non trouvé", "Tarif non trouvé", "Tarif non trouvé"
    
    tarif_base = tarif_ligne.iloc[0]["Prix"]
    taxe = taxe_par_transporteur.get(service.lower(), 0)
    taxe_gasoil = tarif_base * (taxe / 100)
    return tarif_base, taxe_gasoil, tarif_base + taxe_gasoil

def application_calcul_tarif_batch(
    commandes: dd.DataFrame,
    tarifs: dd.DataFrame,
    taxe_par_transporteur: Dict[str, float],
    prix_achat_df: Optional[dd.DataFrame] = None,
    batch_size: int = 1000
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Traite les commandes par lots et parallélise avec ThreadPoolExecutor pour des performances accrues."""
    has_code_pays = "Code Pays" in commandes.columns
    results = []

    def process_batch(batch):
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
        return batch_results

    batches = [commandes.loc[i:i + batch_size - 1].compute() for i in range(0, len(commandes), batch_size)]
    
    # Exécution en parallèle des calculs pour chaque lot
    with ThreadPoolExecutor() as executor:
        for batch_results in executor.map(process_batch, batches):
            results.extend(batch_results)

    # Création du DataFrame final
    commandes_processed = commandes.compute()
    commandes_processed[["Tarif de Base", "Taxe Gasoil", "Tarif Total"]] = pd.DataFrame(results)

    mask_tarif_trouve = commandes_processed["Tarif de Base"] != "Tarif non trouvé"
    commandes_tarifées = commandes_processed[mask_tarif_trouve].copy()
    commandes_sans_tarif = commandes_processed[~mask_tarif_trouve].copy()
    
    if prix_achat_df is not None:
        prix_achat_df = prix_achat_df.compute()
        merge_cols = ["Service de transport"]
        if has_code_pays:
            merge_cols.append("Code Pays")
        commandes_tarifées = commandes_tarifées.merge(
            prix_achat_df, on=merge_cols, how="left"
        ).rename(columns={"Prix Achat": "Prix d'Achat"})
        commandes_tarifées["Marge"] = commandes_tarifées["Tarif Total"] - commandes_tarifées["Prix d'Achat"]
    else:
        commandes_tarifées["Prix d'Achat"] = np.nan
        commandes_tarifées["Marge"] = np.nan

    return commandes_tarifées, commandes_sans_tarif

# Fonctions d'affichage des résultats et des graphiques
def display_results_tab(commandes_tarifées: pd.DataFrame, commandes_sans_tarif: pd.DataFrame):
    """Affiche l'onglet des résultats avec les filtres et les statistiques."""
    st.subheader("Commandes avec Tarifs Calculés")
    st.dataframe(commandes_tarifées, use_container_width=True)
    
    if not commandes_sans_tarif.empty:
        st.subheader("Commandes Sans Tarifs")
        st.dataframe(commandes_sans_tarif, use_container_width=True)

def display_graphics_tab(commandes_tarifées: pd.DataFrame):
    """Affiche des graphiques analytiques."""
    st.subheader("Analyses Graphiques")
    
    chart_type = st.selectbox(
        "Sélectionner le type d'analyse",
        ["Coût par Partenaire", "Coût par Pays", "Coût par Service"]
    )
    
    if chart_type == "Coût par Partenaire":
        data = commandes_tarifées.groupby("Nom du partenaire")["Tarif Total"].agg(
            ['sum', 'mean', 'count']
        ).reset_index()
        fig = px.bar(
            data,
            x="Nom du partenaire",
            y="sum",
            title="Coût total par Partenaire",
            labels={"sum": "Coût total (€)", "Nom du partenaire": "Partenaire"}
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
            labels={"sum": "Coût total (€)", "Pays destination": "Pays"}
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
            labels={"sum": "Coût total (€)", "Service de transport": "Service"}
        )
    
    st.plotly_chart(fig, use_container_width=True)

# Fonctions de téléchargement
def convertir_df_en_excel(df: pd.DataFrame) -> bytes:
    """Convertit un DataFrame en fichier Excel pour le téléchargement."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

def display_download_tab(commandes_tarifées: pd.DataFrame, commandes_sans_tarif: pd.DataFrame):
    """Affiche les options de téléchargement des résultats."""
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

# Fonction principale
def main():
    """Fonction principale de l'application Streamlit."""
    st.title("Application de Calcul des Tarifs de Transport")

    # Paramètres de taxe gasoil
    st.sidebar.subheader("Paramètres de la Taxe Gasoil")
    taxe_par_transporteur = {t: st.sidebar.number_input(f"Taxe {t} (%)", min_value=0.0, max_value=100.0, step=0.1)
                             for t in ["Colissimo", "Chronopost", "DHL", "UPS", "FedEx", "Mondial Relay"]}

    # Téléchargement des fichiers
    with st.expander("Téléchargez les fichiers au format Parquet"):
        commandes_file = st.file_uploader("Fichier des commandes", type=["parquet", "csv"])
        tarifs_file = st.file_uploader("Fichier des tarifs", type=["parquet", "csv"])
        prix_achat_file = st.file_uploader("Fichier des prix d'achat (optionnel)", type=["parquet", "csv"])

    if commandes_file and tarifs_file:
        format_type = "parquet" if commandes_file.name.endswith("parquet") else "csv"
        commandes = load_data(commandes_file, file_format=format_type)
        tarifs = load_data(tarifs_file, file_format=format_type)
        
        prix_achat_df = None
        if prix_achat_file:
            prix_achat_df = load_data(prix_achat_file, file_format=format_type)

        # Calcul des tarifs
        st.write("Calcul des tarifs...")
        commandes_tarifées, commandes_sans_tarif = application_calcul_tarif_batch(
            commandes, tarifs, taxe_par_transporteur, prix_achat_df
        )
        
        # Affichage des résultats
        tabs = st.tabs(["Résultats", "Graphiques", "Téléchargement"])
        with tabs[0]:
            display_results_tab(commandes_tarifées, commandes_sans_tarif)
        with tabs[1]:
            display_graphics_tab(commandes_tarifées)
        with tabs[2]:
            display_download_tab(commandes_tarifées, commandes_sans_tarif)

if __name__ == "__main__":
    main()
