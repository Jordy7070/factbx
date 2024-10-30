import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

# Configuration de la page Streamlit
st.set_page_config(page_title="Calcul des Tarifs de Transport", layout="wide")

def calculer_tarif_avec_taxe(partenaire, service, pays, poids, tarifs, taxe_par_transporteur):
    """Calculate base rate and fuel tax for a shipment."""
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

def application_calcul_tarif(commandes, tarifs, taxe_par_transporteur, prix_achat_df=None):
    """Process orders with pricing calculations."""
    # Vérifier si le Code Pays existe dans les données
    has_code_pays = "Code Pays" in commandes.columns
    
    commandes[["Tarif de Base", "Taxe Gasoil", "Tarif Total"]] = commandes.apply(
        lambda row: calculer_tarif_avec_taxe(
            row["Nom du partenaire"],
            row["Service de transport"],
            row["Pays destination"],
            row["Poids expédition"],
            tarifs,
            taxe_par_transporteur
        ), axis=1, result_type="expand"
    )

    commandes_sans_tarif = commandes[commandes["Tarif de Base"] == "Tarif non trouvé"]
    commandes_tarifées = commandes[commandes["Tarif de Base"] != "Tarif non trouvé"]

    for col in ["Tarif de Base", "Taxe Gasoil", "Tarif Total"]:
        commandes_tarifées[col] = pd.to_numeric(commandes_tarifées[col], errors='coerce')

    if prix_achat_df is not None:
        # Vérifier si le prix d'achat contient Code Pays
        merge_cols = ["Service de transport"]
        if has_code_pays and "Code Pays" in prix_achat_df.columns:
            merge_cols.append("Code Pays")
            
        commandes_tarifées = commandes_tarifées.merge(
            prix_achat_df,
            on=merge_cols,
            how="left"
        )
        
        commandes_tarifées.rename(columns={"Prix Achat": "Prix d'Achat"}, inplace=True)
        commandes_tarifées["Marge"] = commandes_tarifées["Tarif Total"] - commandes_tarifées["Prix d'Achat"]
    else:
        commandes_tarifées["Prix d'Achat"] = np.nan
        commandes_tarifées["Marge"] = np.nan

    return commandes_tarifées, commandes_sans_tarif

def convertir_df_en_excel(df, include_marge=False):
    """Convert DataFrame to Excel file."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        if not include_marge:
            df = df.drop(columns=["Prix d'Achat", "Marge"], errors="ignore")
        df.to_excel(writer, index=False)
    return output.getvalue()

def process_weight_control(compiled_file, facture_file):
    """Process weight control files and return enriched data."""
    compiled_data = pd.read_excel(compiled_file)
    facture_data = pd.read_excel(facture_file)
    
    compiled_data = compiled_data.rename(columns={
        "Numéro de commande partenaire": "Tracking",
        "Poids expédition": "Poids_expédition"
    })
    facture_data = facture_data.rename(columns={
        "Poids constaté": "Poids_facture"
    })
    
    compiled_data["Tracking"] = compiled_data["Tracking"].astype(str)
    facture_data["Tracking"] = facture_data["Tracking"].astype(str)
    
    merged_data = pd.merge(facture_data, compiled_data, on="Tracking", how="left")
    merged_data["Ecart_Poids"] = merged_data["Poids_facture"] - merged_data["Poids_expédition"]
    
    return merged_data

def display_results_tab(commandes_tarifées, commandes_sans_tarif):
    """Display filtered results for both priced and unpriced orders."""
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
            total = commandes_tarifées['Tarif Total'].sum()
            st.metric("Montant total", f"{total:.2f} €")
    
    filtered_data = commandes_tarifées.copy()
    for col, selected in filters.items():
        if selected:
            col_name = "Nom du partenaire" if col == "Partenaire" else \
                      "Service de transport" if col == "Service" else \
                      "Pays destination"
            filtered_data = filtered_data[filtered_data[col_name].isin(selected)]
    
    st.subheader("Commandes avec Tarifs Calculés")
    display_columns = ["Nom du partenaire", "Service de transport", 
                      "Pays destination", "Poids expédition",
                      "Tarif de Base", "Taxe Gasoil", "Tarif Total"]
    
    if "Code Pays" in filtered_data.columns:
        display_columns.insert(3, "Code Pays")
    
    st.dataframe(filtered_data[display_columns])

    st.subheader("Commandes Sans Tarifs")
    if not commandes_sans_tarif.empty:
        display_columns_sans_tarif = ["Nom du partenaire", "Service de transport", 
                                    "Pays destination", "Poids expédition"]
        if "Code Pays" in commandes_sans_tarif.columns:
            display_columns_sans_tarif.insert(3, "Code Pays")
            
        st.dataframe(commandes_sans_tarif[display_columns_sans_tarif])
        st.warning(f"Nombre de commandes sans tarif trouvé : {len(commandes_sans_tarif)}")
    else:
        st.success("Toutes les commandes ont un tarif calculé")

def display_graphics_tab(commandes_tarifées):
    """Display interactive graphics."""
    st.subheader("Analyses Graphiques")
    
    chart_type = st.selectbox(
        "Sélectionner le type d'analyse",
        ["Coût par Partenaire", "Coût par Pays", "Coût par Service"]
    )
    
    if not commandes_tarifées.empty:
        if chart_type == "Coût par Partenaire":
            data = commandes_tarifées.groupby("Nom du partenaire")["Tarif Total"].sum().reset_index()
            fig = px.bar(
                data,
                x="Nom du partenaire",
                y="Tarif Total",
                title="Coût total par Partenaire",
                labels={"Tarif Total": "Coût total (€)", "Nom du partenaire": "Partenaire"}
            )
        elif chart_type == "Coût par Pays":
            data = commandes_tarifées.groupby("Pays destination")["Tarif Total"].sum().reset_index()
            fig = px.bar(
                data,
                x="Pays destination",
                y="Tarif Total",
                title="Coût total par Pays",
                labels={"Tarif Total": "Coût total (€)", "Pays destination": "Pays"}
            )
        else:
            data = commandes_tarifées.groupby("Service de transport")["Tarif Total"].sum().reset_index()
            fig = px.bar(
                data,
                x="Service de transport",
                y="Tarif Total",
                title="Coût total par Service",
                labels={"Tarif Total": "Coût total (€)", "Service de transport": "Service"}
            )
        
        st.plotly_chart(fig)

def display_download_tab(commandes_tarifées, commandes_sans_tarif):
    """Display download buttons."""
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

def display_analysis_tab(commandes_tarifées):
    """Display margin analysis."""
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
        
        st.dataframe(marge_pays)
        
        fig = px.bar(
            marge_pays,
            x="Code Pays",
            y="Marge Totale",
            title="Marge Totale par Pays",
            labels={"Marge Totale": "Marge (€)", "Code Pays": "Pays"}
        )
        st.plotly_chart(fig)
    
    st.subheader("Détail des Marges")
    display_columns = [
        "Nom du partenaire", "Service de transport",
        "Pays destination", "Poids expédition",
        "Tarif Total", "Prix d'Achat", "Marge"
    ]
    if "Code Pays" in commandes_tarifées.columns:
        display_columns.insert(3, "Code Pays")
        
    st.dataframe(commandes_tarifées[display_columns])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Marge Totale", f"{commandes_tarifées['Marge'].sum():.2f} €")
    with col2:
        st.metric("Marge Moyenne", f"{commandes_tarifées['Marge'].mean():.2f} €")
    with col3:
        st.metric("Nombre de Commandes", len(commandes_tarifées))

def display_weight_control_tab(compiled_file, facture_file):
    """Display weight control analysis."""
    merged_data = process_weight_control(compiled_file, facture_file)
    
    st.subheader("Analyse des Écarts de Poids")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Écart Moyen", f"{merged_data['Ecart_Poids'].mean():.2f} kg")
    with col2:
        st.metric("Écart Total", f"{merged_data['Ecart_Poids'].sum():.2f} kg")
    
    st.download_button(
        "📥 Télécharger rapport de contrôle",
        convertir_df_en_excel(merged_data),
        "controle_poids_enrichi.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    st.subheader("Détail des Écarts")
    st.dataframe(merged_data)

def main():
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
            step=0.1
        )
        for transporteur in mots_cles_transporteurs
    }
    
    # File uploaders
    st.subheader("Téléchargement des Fichiers")
    files = {
        "commandes": st.file_uploader("Fichier de commandes", type=["xlsx"]),
        "tarifs": st.file_uploader("Fichier de tarifs", type=["xlsx"]),
        "prix_achat": st.file_uploader("Fichier prix d'achat (optionnel)", type=["xlsx"]),
        "compiled": st.file_uploader("Fichier compilé", type=["xlsx"]),
        "facture": st.file_uploader("Fichier de facturation", type=["xlsx"])
    }
    
    if files["commandes"] and files["tarifs"]:
        try:
            # Load and process main data
            commandes = pd.read_excel(files["commandes"])
            tarifs = pd.read_excel(files["tarifs"])
            prix_achat_df = None
            
            # Vérification des colonnes requises dans le fichier de commandes
            required_columns_commandes = ["Nom du partenaire", "Service de transport", 
                                       "Pays destination", "Poids expédition"]
            if not all(col in commandes.columns for col in required_columns_commandes):
                missing_cols = [col for col in required_columns_commandes if col not in commandes.columns]
                st.error(f"Colonnes manquantes dans le fichier de commandes: {', '.join(missing_cols)}")
                return

            # Vérification des colonnes requises dans le fichier de tarifs
            required_columns_tarifs = ["Partenaire", "Service", "Pays", "PoidsMin", "PoidsMax", "Prix"]
            if not all(col in tarifs.columns for col in required_columns_tarifs):
                missing_cols = [col for col in required_columns_tarifs if col not in tarifs.columns]
                st.error(f"Colonnes manquantes dans le fichier de tarifs: {', '.join(missing_cols)}")
                return
            
            if files["prix_achat"]:
                try:
                    prix_achat_df = pd.read_excel(files["prix_achat"])
                    required_columns = ["Service de transport", "Prix Achat"]
                    if not all(col in prix_achat_df.columns for col in required_columns):
                        st.error("Le fichier de prix d'achat doit contenir les colonnes 'Service de transport' et 'Prix Achat'")
                        prix_achat_df = None
                except Exception as e:
                    st.error(f"Erreur lors de la lecture du fichier de prix d'achat: {str(e)}")
                    prix_achat_df = None
            
            # Calculate prices and margins
            commandes_tarifées, commandes_sans_tarif = application_calcul_tarif(
                commandes, tarifs, taxe_par_transporteur, prix_achat_df
            )
            
            # Create tabs for different views
            tabs = st.tabs([
                "📊 Résultats", 
                "📈 Graphiques", 
                "💾 Téléchargement", 
                "💰 Analyse", 
                "⚖️ Contrôle"
            ])
            
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
                if files["compiled"] and files["facture"]:
                    display_weight_control_tab(files["compiled"], files["facture"])
                else:
                    st.info("Veuillez télécharger les fichiers compilé et de facturation pour le contrôle de poids.")
                    
        except Exception as e:
            st.error(f"Une erreur s'est produite lors du traitement des données: {str(e)}")
            st.info("Veuillez vérifier le format des fichiers et réessayer.")
            
    else:
        st.info("Veuillez télécharger les fichiers de commandes et de tarifs pour commencer l'analyse.")

if __name__ == "__main__":
    main()