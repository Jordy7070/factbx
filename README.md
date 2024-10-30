# Application de Calcul des Tarifs de Transport

Application Streamlit pour le calcul et l'analyse des tarifs de transport avec gestion de la taxe gasoil.

## Installation

1. Cloner le repository
```bash
git clone https://github.com/Jordy7070/factbx.git
```

2. Installer les dépendances
```bash
pip install -r requirements.txt
```

3. Lancer l'application
```bash
streamlit run app.py
```

## Fichiers requis

### Fichier de commandes
- Nom du partenaire
- Service de transport
- Pays destination
- Poids expédition
- Code Pays (optionnel)

### Fichier de tarifs
- Partenaire
- Service
- Pays
- PoidsMin
- PoidsMax
- Prix

### Fichier de prix d'achat (optionnel)
- Service de transport
- Prix Achat
- Code Pays (optionnel)
