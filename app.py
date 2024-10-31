import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from typing import Tuple, Dict, Optional
import plotly.express as px

# Configuration de la page
st.set_page_config(
    page_title="Calcul des Tarifs de Transport et Contrôle de Poids",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data(file_uploader, usecols: Optional[list] = None) -> pd.DataFrame:
    """Charge et met en cache les données depuis un fichier avec vérification des colonnes."""
    try:
        data = pd.read_excel(file_uploader, usecols=usecols)
        return data
    except ValueError as e:
        st.error(f"Erreur de chargement des données : {e}")
        raise

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
    
    # Calcul de la marge en fonction de la tranche de poids appropriée
    if prix_achat_df is not None:
        commandes_tarifées = commandes_tarifées.merge(
            prix_achat_df,
            left_on=["Service de transport", "Pays destination"],
            right_on=["Service", "Pays"],
            how="left"
        )
        commandes_tarifées = commandes_tarifées[
            (commandes_tarifées["Poids expédition"] >= commandes_tarifées["PoidsMin"]) &
            (commandes_tarifées["Poids expédition"] <= commandes_tarifées["PoidsMax"])
        ]
        commandes_tarifées["Marge"] = commandes_tarifées["Tarif Total"] - commandes_tarifées["Prix d'Achat"]
    else:
        commandes_tarifées["Prix d'Achat"] = np.nan
        commandes_tarifées["Marge"] = np.nan

    return commandes_tarifées, commandes_sans_tarif

@st.cache_data
def process_weight_control(compiled_file, facture_file) -> pd.DataFrame:
    """Analyse les écarts de poids entre les données compilées et facturées avec gestion des unités de poids."""
    compiled_data = load_data(compiled_file, usecols=["Numéro de tracking", "Poids expédition"])
    facture_data = load_data(facture_file, usecols=["Tracking", "Poids constaté"])

    compiled_data = compiled_data.rename(columns={"Numéro de tracking": "Tracking", "Poids expédition": "Poids_expédition"})
    facture_data = facture_data.rename(columns={"Poids constaté": "Poids_facture"})

    # Conversion des poids du fichier compilé de grammes à kilogrammes
    compiled_data["Poids_expédition"] = compiled_data["Poids_expédition"] / 1000

    # Fusion et calcul de l'écart de poids
    merged_data = pd.merge(facture_data, compiled_data, on="Tracking", how="left")
    merged_data["Ecart_Poids"] = merged_data["Poids_facture"] - merged_data["Poids_expédition"]

    return merged_data

def convertir_df_en_excel(df: pd.DataFrame) -> bytes:
    """Convertit un DataFrame en fichier Excel pour le téléchargement."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

def display_results_tab(commandes_tarifées: pd.DataFrame, commandes_sans_tarif: pd.DataFrame):
    st.subheader("Commandes avec Tarifs Calculés")
    st.dataframe(commandes_tarifées, height=400, use_container_width=True)
    
    if not commandes_sans_tarif.empty:
        st.subheader("Commandes Sans Tarifs")
        st.dataframe(commandes_sans_tarif, height=200, use_container_width=True)

    # Boutons de téléchargement
    st.download_button(
        label="📥 Télécharger commandes avec tarifs",
        data=convertir_df_en_excel(commandes_tarifées),
        file_name="commandes_avec_tarifs.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_commandes_tarifées"
    )
    st.download_button(
        label="📥 Télécharger commandes sans tarif",
        data=convertir_df_en_excel(commandes_sans_tarif),
        file_name="commandes_sans_tarifs.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_commandes_sans_tarif"
    )

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

def display_weight_control_tab(compiled_file, facture_file):
    merged_data = process_weight_control(compiled_file, facture_file)
    st.subheader("Analyse des Écarts de Poids")
    st.dataframe(merged_data, height=400, use_container_width=True)

    # Bouton de téléchargement pour les écarts de poids
    st.download_button(
        label="📥 Télécharger le rapport d'écarts de poids",
        data=convertir_df_en_excel(merged_data),
        file_name="rapport_ecarts_poids.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_ecarts_poids"
    )

def display_analysis_tab(commandes_tarifées: pd.DataFrame):
    """Affiche une analyse des marges pour les commandes tarifées."""
    st.subheader("Analyse de Marge")
    
    total_marge = commandes_tarifées["Marge"].sum()
    marge_moyenne = commandes_tarifées["Marge"].mean()
    # Affiche les statistiques principales
    col1, col2, col3 = st.columns(3)
    col1.metric("Marge Totale", f"{total_marge:.2f} €")
    col2.metric("Marge Moyenne", f"{marge_moyenne:.2f} €")
    col3.metric("Nombre de Commandes", len(commandes_tarifées))

    # Détail des marges par pays
    marge_par_pays = commandes_tarifées.groupby("Pays destination")["Marge"].sum().reset_index()
    fig = px.bar(marge_par_pays, x="Pays destination", y="Marge", title="Marge Totale par Pays")
    st.plotly_chart(fig, use_container_width=True)

    # Détail des marges par partenaire
    marge_par_partenaire = commandes_tarifées.groupby("Nom du partenaire")["Marge"].sum().reset_index()
    fig2 = px.bar(marge_par_partenaire, x="Nom du partenaire", y="Marge", title="Marge Totale par Partenaire")
    st.plotly_chart(fig2, use_container_width=True)

def main():
    st.title("Calcul des Tarifs de Transport & Contrôle des Écarts de Poids")
    st.markdown("Cette application permet de calculer les tarifs de transport, d'analyser les écarts de poids et de générer des rapports.")

    # Téléchargement des fichiers
    st.sidebar.title("Paramètres")
    st.sidebar.write("Téléchargez les fichiers requis ci-dessous pour démarrer :")
    commandes_file = st.sidebar.file_uploader("Fichier de commandes", type=["xlsx"])
    tarifs_file = st.sidebar.file_uploader("Fichier de tarifs", type=["xlsx"])
    prix_achat_file = st.sidebar.file_uploader("Fichier prix d'achat (optionnel)", type=["xlsx"])
    compiled_file = st.sidebar.file_uploader("Fichier compilé", type=["xlsx"])
    facture_file = st.sidebar.file_uploader("Fichier de facturation", type=["xlsx"])

    # Section de configuration des taxes
    st.sidebar.subheader("Configuration des Taxes par Transporteur")
    taxe_par_transporteur = {
        transporteur: st.sidebar.number_input(f"Taxe {transporteur} (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
        for transporteur in ["Colissimo", "Chronopost", "DHL", "UPS", "FedEx", "Mondial Relay"]
    }

    # Vérification de la disponibilité des fichiers nécessaires
    if commandes_file and tarifs_file and compiled_file and facture_file:
        try:
            commandes = load_data(commandes_file, usecols=["Nom du partenaire", "Service de transport", "Pays destination", "Poids expédition"])
            tarifs = load_data(tarifs_file, usecols=["Partenaire", "Service", "Pays", "PoidsMin", "PoidsMax", "Prix"])
            prix_achat_df = load_data(prix_achat_file) if prix_achat_file else None

            # Calcul des tarifs pour les commandes
            commandes_tarifées, commandes_sans_tarif = application_calcul_tarif_batch(commandes, tarifs, taxe_par_transporteur, prix_achat_df)

            st.success("Fichiers chargés et analyses prêtes.")
            tabs = st.tabs(["📊 Résultats", "📈 Graphiques", "💾 Téléchargement", "💰 Analyse", "⚖️ Contrôle de Poids"])
            with tabs[0]:
                display_results_tab(commandes_tarifées, commandes_sans_tarif)
            with tabs[1]:
                display_graphics_tab(commandes_tarifées)
            with tabs[2]:
                st.write("Téléchargement des fichiers est disponible dans les onglets spécifiques.")
            with tabs[3]:
                display_analysis_tab(commandes_tarifées)
            with tabs[4]:
                if compiled_file and facture_file:
                    display_weight_control_tab(compiled_file, facture_file)
                else:
                    st.info("Veuillez télécharger les fichiers compilé et de facturation pour le contrôle de poids.")

        except Exception as e:
            st.error(f"Une erreur s'est produite : {str(e)}")
    else:
        st.info("Veuillez télécharger tous les fichiers nécessaires pour démarrer l'analyse.")

if __name__ == "__main__":
    main()
