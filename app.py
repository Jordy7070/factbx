import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from typing import Tuple, Dict, Optional, List
import plotly.express as px
from plotly.graph_objects import Figure
import plotly.graph_objects as go
from datetime import datetime

# Constantes globales
SUPPORTED_CARRIERS = ["Colissimo", "Chronopost", "DHL", "UPS", "FedEx", "Mondial Relay"]
DEFAULT_TAX_RATE = 0.0
BATCH_SIZE = 1000

class ConfigManager:
    """Gestion de la configuration de l'application"""
    @staticmethod
    def initialize_config():
        st.set_page_config(
            page_title="Calculateur de Tarifs Transport",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'About': "Calculateur de tarifs de transport v2.0",
                'Get Help': 'https://www.example.com/help',
                'Report a bug': "mailto:support@example.com"
            }
        )

    @staticmethod
    def setup_sidebar() -> Dict[str, float]:
        with st.sidebar:
            st.title("⚙️ Configuration")
            st.markdown("---")
            
            # Ajout d'un sélecteur de thème
            theme = st.selectbox(
                "🎨 Thème",
                ["Light", "Dark"],
                help="Choisissez le thème de l'interface"
            )
            
            st.markdown("### 🚛 Taxes par Transporteur")
            # Permet la saisie groupée des taxes
            bulk_tax = st.number_input(
                "Appliquer la même taxe à tous les transporteurs",
                min_value=0.0,
                max_value=100.0,
                value=DEFAULT_TAX_RATE,
                step=0.1,
                help="Définir rapidement la même taxe pour tous les transporteurs"
            )
            
            if st.button("Appliquer à tous"):
                return {carrier.lower(): bulk_tax for carrier in SUPPORTED_CARRIERS}
            
            st.markdown("#### Taxes individuelles")
            return {
                carrier.lower(): st.number_input(
                    f"Taxe {carrier} (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=DEFAULT_TAX_RATE,
                    step=0.1,
                    help=f"Taxe spécifique pour {carrier}"
                )
                for carrier in SUPPORTED_CARRIERS
            }

class DataLoader:
    """Gestion du chargement et de la validation des données"""
    @staticmethod
    @st.cache_data
    def load_data(file_uploader, usecols: Optional[list] = None, sheet_name: Optional[str] = None) -> pd.DataFrame:
        try:
            if sheet_name:
                data = pd.read_excel(file_uploader, sheet_name=sheet_name)
            else:
                data = pd.read_excel(file_uploader)
            
            if usecols:
                missing_cols = [col for col in usecols if col not in data.columns]
                if missing_cols:
                    raise ValueError(f"Colonnes manquantes : {', '.join(missing_cols)}")
                data = data[usecols]
            
            return data
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier : {str(e)}")
            return pd.DataFrame()

class TariffCalculator:
    """Calcul des tarifs et analyses"""
    @staticmethod
    @st.cache_data
    def calculate_tariff(
        partenaire: str,
        service: str, 
        pays: str,
        poids: float,
        tarifs: pd.DataFrame,
        taxe_par_transporteur: Dict[str, float]
    ) -> Tuple[float, float, float]:
        try:
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
        except Exception as e:
            st.error(f"Erreur de calcul : {str(e)}")
            return "Erreur", "Erreur", "Erreur"

    @staticmethod
    @st.cache_data
    def process_orders_batch(
        commandes: pd.DataFrame,
        tarifs: pd.DataFrame,
        taxe_par_transporteur: Dict[str, float],
        prix_achat_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        total_batches = len(commandes) // BATCH_SIZE + 1
        
        for batch_num, start_idx in enumerate(range(0, len(commandes), BATCH_SIZE)):
            status_text.text(f"Traitement du lot {batch_num + 1}/{total_batches}...")
            batch = commandes.iloc[start_idx:start_idx + BATCH_SIZE]
            
            batch_results = [
                TariffCalculator.calculate_tariff(
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
            progress_bar.progress(min((batch_num + 1) / total_batches, 1.0))

        progress_bar.empty()
        status_text.empty()

        return TariffCalculator._process_results(commandes, results, prix_achat_df)

    @staticmethod
    def _process_results(
        commandes: pd.DataFrame,
        results: List[Tuple[float, float, float]],
        prix_achat_df: Optional[pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        commandes_processed = commandes.copy()
        commandes_processed[["Tarif de Base", "Taxe Gasoil", "Tarif Total"]] = pd.DataFrame(results)
        
        mask_tarif_trouve = commandes_processed["Tarif de Base"] != "Tarif non trouvé"
        commandes_tarifées = commandes_processed[mask_tarif_trouve].copy()
        commandes_sans_tarif = commandes_processed[~mask_tarif_trouve].copy()
        
        numeric_columns = ["Tarif de Base", "Taxe Gasoil", "Tarif Total"]
        commandes_tarifées[numeric_columns] = commandes_tarifées[numeric_columns].apply(pd.to_numeric, errors="coerce")
        
        if prix_achat_df is not None:
            commandes_tarifées = TariffCalculator._add_margin_calculation(commandes_tarifées, prix_achat_df)
        else:
            commandes_tarifées["Prix d'Achat"] = np.nan
            commandes_tarifées["Marge"] = np.nan
            
        return commandes_tarifées, commandes_sans_tarif

    @staticmethod
    def _add_margin_calculation(
        commandes_tarifées: pd.DataFrame,
        prix_achat_df: pd.DataFrame
    ) -> pd.DataFrame:
        merged_df = commandes_tarifées.merge(
            prix_achat_df,
            left_on=["Service de transport", "Pays destination"],
            right_on=["Service", "Pays"],
            how="left"
        )
        
        valid_weights = (
            (merged_df["Poids expédition"] >= merged_df["PoidsMin"]) &
            (merged_df["Poids expédition"] <= merged_df["PoidsMax"])
        )
        
        result_df = merged_df[valid_weights].copy()
        result_df["Marge"] = result_df["Tarif Total"] - result_df["Prix d'Achat"]
        
        return result_df

class UIManager:
    """Gestion de l'interface utilisateur"""
    @staticmethod
    def display_header():
        st.title("🚚 Calculateur de Tarifs Transport")
        st.markdown("""
        Cette application permet de :
        - Calculer les tarifs de transport
        - Analyser les écarts de poids
        - Visualiser les statistiques
        - Générer des rapports détaillés
        """)

    @staticmethod
    def display_file_uploaders():
        with st.expander("📁 Chargement des fichiers", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                commandes_file = st.file_uploader(
                    "Fichier de commandes (obligatoire)",
                    type=["xlsx"],
                    help="Fichier Excel contenant les commandes à traiter"
                )
                tarifs_file = st.file_uploader(
                    "Fichier de tarifs (obligatoire)",
                    type=["xlsx"],
                    help="Fichier Excel contenant la grille tarifaire"
                )
                prix_achat_file = st.file_uploader(
                    "Fichier prix d'achat (optionnel)",
                    type=["xlsx"],
                    help="Fichier Excel contenant les prix d'achat"
                )
            
            with col2:
                compiled_file = st.file_uploader(
                    "Fichier compilé (contrôle poids)",
                    type=["xlsx"],
                    help="Fichier Excel contenant les données compilées"
                )
                facture_file = st.file_uploader(
                    "Fichier de facturation (contrôle poids)",
                    type=["xlsx"],
                    help="Fichier Excel contenant les données de facturation"
                )
            
            return commandes_file, tarifs_file, prix_achat_file, compiled_file, facture_file

    @staticmethod
    def create_plotly_figure(
        data: pd.DataFrame,
        x: str,
        y: str,
        title: str,
        kind: str = "bar"
    ) -> Figure:
        if kind == "bar":
            fig = go.Figure(data=[
                go.Bar(
                    x=data[x],
                    y=data[y],
                    text=data[y].round(2),
                    textposition='auto',
                )
            ])
        else:
            fig = go.Figure(data=[
                go.Scatter(
                    x=data[x],
                    y=data[y],
                    mode='lines+markers',
                    text=data[y].round(2),
                    textposition='top center',
                )
            ])

        fig.update_layout(
            title=title,
            xaxis_title=x,
            yaxis_title=y,
            template="plotly_white",
            showlegend=False
        )
        
        return fig

    @staticmethod
    def display_results_tab(commandes_tarifées: pd.DataFrame, commandes_sans_tarif: pd.DataFrame):
        st.subheader("📊 Résultats des Calculs")
        
        # Métriques principales
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Commandes Traitées",
                f"{len(commandes_tarifées):,}",
                f"{len(commandes_tarifées) / (len(commandes_tarifées) + len(commandes_sans_tarif)) * 100:.1f}%"
            )
        with col2:
            st.metric(
                "Montant Total",
                f"{commandes_tarifées['Tarif Total'].sum():,.2f} €"
            )
        with col3:
            st.metric(
                "Commandes Sans Tarif",
                f"{len(commandes_sans_tarif):,}",
                delta_color="inverse"
            )

        # Affichage des données
        tabs = st.tabs(["🎯 Commandes Tarifées", "⚠️ Commandes Sans Tarif"])
        with tabs[0]:
            st.dataframe(
                commandes_tarifées,
                use_container_width=True,
                height=400,
                column_config={
                    "Tarif Total": st.column_config.NumberColumn(
                        "Tarif Total",
                        format="%.2f €"
                    ),
                    "Marge": st.column_config.NumberColumn(
                        "Marge",
                        format="%.2f €"
                    )
                }
            )
            
        with tabs[1]:
            if not commandes_sans_tarif.empty:
                st.dataframe(commandes_sans_tarif, use_container_width=True, height=200)
            else:
                st.success("Toutes les commandes ont été tarifées avec succès! 🎉")

        # Export
        col1, col2 = st.columns(2)
        with col1:
            UIManager._download_button(
                commandes_tarifées,
                "commandes_tarifees.xlsx",
                "💾 Télécharger commandes tarifées"
            )
        with col2:
            if not commandes_sans_tarif.empty:
                UIManager._download_button(
                    commandes_sans_tarif,
                    "commandes_sans_tarif.xlsx",
                    "⚠️ Télécharger commandes sans tarif"
                )

    @staticmethod
    def _download_button(df: pd.DataFrame, filename: str, label: str):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
            writer.sheets['Sheet1'].set_column('A:Z', 15)  # Ajuste la largeur des colonnes
        
        st.download_button(
            label=label,
            data=output.getvalue(),
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    @staticmethod
    def display_graphics_tab(commandes_tarifées: pd.DataFrame):
        st.subheader("📈 Analyses Graphiques")
        
        # Configuration des analyses
        col1, col2 = st.columns([2, 1])
        with col1:
            chart_type = st.selectbox(
                "Type d'analyse",
                ["Coût par Partenaire", "Coût par Pays", "Coût par Service", "Évolution des Marges"]
            )
        with col2:
            display_mode = st.radio(
                "Mode d'affichage",
                ["Barres", "Ligne"],
                horizontal=True
            )

        # Préparation des données selon le type d'analyse
        if chart_type == "Coût par Partenaire":
            data = commandes_tarifées.groupby("Nom du partenaire").agg({
                "Tarif Total": "sum",
                "Marge": "sum"
            }).reset_index()
            fig1 = UIManager.create_plotly_figure(
                data,
                "Nom du partenaire",
                "Tarif Total",
                "Coût total par Partenaire",
                display_mode.lower()
            )
            fig2 = UIManager.create_plotly_figure(
                data,
                "Nom du partenaire",
                "Marge",
                "Marge par Partenaire",
                display_mode.lower()
            )
        elif chart_type == "Coût par Pays":
            data = commandes_tarifées.groupby("Pays destination").agg({
                "Tarif Total": "sum",
                "Marge": "sum"
            }).reset_index()
            fig1 = UIManager.create_plotly_figure(
                data,
                "Pays destination",
                "Tarif Total",
                "Coût total par Pays",
                display_mode.lower()
            )
            fig2 = UIManager.create_plotly_figure(
                data,
                "Pays destination",
                "Marge",
                "Marge par Pays",
                display_mode.lower()
            )
        elif chart_type == "Coût par Service":
            data = commandes_tarifées.groupby("Service de transport").agg({
                "Tarif Total": "sum",
                "Marge": "sum"
            }).reset_index()
            fig1 = UIManager.create_plotly_figure(
                data,
                "Service de transport",
                "Tarif Total",
                "Coût total par Service",
                display_mode.lower()
            )
            fig2 = UIManager.create_plotly_figure(
                data,
                "Service de transport",
                "Marge",
                "Marge par Service",
                display_mode.lower()
            )
        else:  # Évolution des Marges
            commandes_tarifées['Date'] = pd.to_datetime('today')  # À remplacer par la vraie colonne de date
            data = commandes_tarifées.groupby('Date').agg({
                "Tarif Total": "sum",
                "Marge": "sum"
            }).reset_index()
            fig1 = UIManager.create_plotly_figure(
                data,
                "Date",
                "Tarif Total",
                "Évolution des coûts",
                "ligne"
            )
            fig2 = UIManager.create_plotly_figure(
                data,
                "Date",
                "Marge",
                "Évolution des marges",
                "ligne"
            )

        # Affichage des graphiques
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            st.plotly_chart(fig2, use_container_width=True)

        # Export des données
        if st.button("💾 Exporter les données d'analyse"):
            UIManager._download_button(
                data,
                f"analyse_{chart_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                "📥 Télécharger les données"
            )

    @staticmethod
    def display_weight_control_tab(compiled_file, facture_file):
        st.subheader("⚖️ Contrôle des Écarts de Poids")
        
        try:
            # Chargement et traitement des données
            merged_data = WeightController.process_weight_control(compiled_file, facture_file)
            
            # Affichage des métriques principales
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Nombre total d'envois",
                    f"{len(merged_data):,}"
                )
            with col2:
                ecart_moyen = merged_data["Ecart_Poids"].mean()
                st.metric(
                    "Écart moyen",
                    f"{ecart_moyen:.2f} kg",
                    delta=f"{ecart_moyen/merged_data['Poids_expédition'].mean()*100:.1f}%"
                )
            with col3:
                ecarts_significatifs = merged_data[abs(merged_data["Ecart_Poids"]) > 0.5]
                st.metric(
                    "Écarts significatifs",
                    f"{len(ecarts_significatifs):,}",
                    f"{len(ecarts_significatifs)/len(merged_data)*100:.1f}%",
                    delta_color="inverse"
                )

            # Affichage des données avec filtres
            st.subheader("Détail des écarts")
            col1, col2 = st.columns([1, 3])
            with col1:
                filter_type = st.selectbox(
                    "Filtre",
                    ["Tous les écarts", "Écarts significatifs (>0.5kg)", "Écarts positifs", "Écarts négatifs"]
                )
            
            # Application des filtres
            if filter_type == "Écarts significatifs (>0.5kg)":
                filtered_data = merged_data[abs(merged_data["Ecart_Poids"]) > 0.5]
            elif filter_type == "Écarts positifs":
                filtered_data = merged_data[merged_data["Ecart_Poids"] > 0]
            elif filter_type == "Écarts négatifs":
                filtered_data = merged_data[merged_data["Ecart_Poids"] < 0]
            else:
                filtered_data = merged_data

            # Affichage du tableau filtré
            st.dataframe(
                filtered_data,
                use_container_width=True,
                height=400,
                column_config={
                    "Poids_expédition": st.column_config.NumberColumn(
                        "Poids expédié",
                        format="%.2f kg"
                    ),
                    "Poids_facture": st.column_config.NumberColumn(
                        "Poids facturé",
                        format="%.2f kg"
                    ),
                    "Ecart_Poids": st.column_config.NumberColumn(
                        "Écart",
                        format="%.2f kg"
                    )
                }
            )

            # Graphique des écarts
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=merged_data["Ecart_Poids"],
                nbinsx=50,
                name="Distribution des écarts"
            ))
            fig.update_layout(
                title="Distribution des écarts de poids",
                xaxis_title="Écart (kg)",
                yaxis_title="Nombre d'envois",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            # Export des données
            UIManager._download_button(
                filtered_data,
                f"rapport_ecarts_poids_{datetime.now().strftime('%Y%m%d')}.xlsx",
                "💾 Télécharger le rapport d'écarts"
            )

        except Exception as e:
            st.error(f"Erreur lors du traitement des données : {str(e)}")

class WeightController:
    """Gestion du contrôle des poids"""
    @staticmethod
    @st.cache_data
    def process_weight_control(compiled_file, facture_file) -> pd.DataFrame:
        # Chargement des données
        compiled_data = DataLoader.load_data(
            compiled_file,
            usecols=["Numéro de tracking", "Poids expédition"]
        )
        facture_data = DataLoader.load_data(
            facture_file,
            usecols=["Tracking", "Poids constaté"]
        )

        # Renommage des colonnes
        compiled_data = compiled_data.rename(columns={
            "Numéro de tracking": "Tracking",
            "Poids expédition": "Poids_expédition"
        })
        facture_data = facture_data.rename(columns={
            "Poids constaté": "Poids_facture"
        })

        # Conversion des poids en kg
        compiled_data["Poids_expédition"] = compiled_data["Poids_expédition"] / 1000

        # Fusion et calcul des écarts
        merged_data = pd.merge(
            facture_data,
            compiled_data,
            on="Tracking",
            how="left"
        )
        merged_data["Ecart_Poids"] = merged_data["Poids_facture"] - merged_data["Poids_expédition"]

        return merged_data

def main():
    """Point d'entrée principal de l'application"""
    try:
        # Initialisation
        ConfigManager.initialize_config()
        taxe_par_transporteur = ConfigManager.setup_sidebar()
        
        # Interface principale
        UIManager.display_header()
        files = UIManager.display_file_uploaders()
        commandes_file, tarifs_file, prix_achat_file, compiled_file, facture_file = files

        if commandes_file and tarifs_file:
            # Chargement des données
            commandes = DataLoader.load_data(
                commandes_file,
                usecols=["Nom du partenaire", "Service de transport", "Pays destination", "Poids expédition"]
            )
            tarifs = DataLoader.load_data(
                tarifs_file,
                usecols=["Partenaire", "Service", "Pays", "PoidsMin", "PoidsMax", "Prix"]
            )
            prix_achat_df = DataLoader.load_data(prix_achat_file) if prix_achat_file else None

            # Calcul des tarifs
            commandes_tarifées, commandes_sans_tarif = TariffCalculator.process_orders_batch(
                commandes,
                tarifs,
                taxe_par_transporteur,
                prix_achat_df
            )

            # Affichage des résultats
            tabs = st.tabs([
                "📊 Résultats",
                "📈 Graphiques",
                "💰 Marges",
                "⚖️ Contrôle Poids",
                "📋 Export"
            ])

            with tabs[0]:
                UIManager.display_results_tab(commandes_tarifées, commandes_sans_tarif)
            with tabs[1]:
                UIManager.display_graphics_tab(commandes_tarifées)
            with tabs[2]:
                UIManager.display_analysis_tab(commandes_tarifées)
            with tabs[3]:
                if compiled_file and facture_file:
                    UIManager.display_weight_control_tab(compiled_file, facture_file)
                else:
                    st.info("📤 Veuillez charger les fichiers de contrôle de poids.")
            with tabs[4]:
                st.info("💾 Les options d'export sont disponibles dans chaque onglet spécifique.")

    except Exception as e:
        st.error(f"Une erreur s'est produite : {str(e)}")
        st.info("🔄 Veuillez vérifier vos données et réessayer.")

if __name__ == "__main__":
    main()
