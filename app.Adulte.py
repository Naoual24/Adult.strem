import streamlit as st
import joblib
import pandas as pd
import zipfile
import os

# Chemin vers le fichier zip
ZIP_PATH = "adulte_model.zip"
EXTRACT_PATH = "models/"

# Créer le dossier d'extraction s'il n'existe pas
os.makedirs(EXTRACT_PATH, exist_ok=True)

# Décompresser le fichier zip s'il n'a pas encore été extrait
MODEL_PATH = os.path.join(EXTRACT_PATH, "adulte.joblib")

if not os.path.exists(MODEL_PATH):
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)
        st.success("Fichiers décompressés avec succès !")
    except Exception as e:
        st.error(f"Erreur lors de la décompression : {e}")
else:
    st.info("Le fichier modèle a déjà été extrait.")

# Charger le modèle
@st.cache_resource
def charger_modele():
    try:
        modele = joblib.load(MODEL_PATH)
        return modele
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

modele = charger_modele()

# Récupérer les colonnes exactes utilisées pour l'entraînement
def obtenir_colonnes_model(modele):
    # Récupérer les colonnes du modèle (ceci est spécifique à votre modèle, selon la méthode d'entraînement)
    if hasattr(modele, 'feature_importances_'):  # Vérifier si le modèle a été formé
        colonnes = modele.feature_importances_
        # Ce code est un exemple ; vous devez récupérer les bonnes colonnes selon le type de modèle
        return colonnes
    return []

# Assurez-vous de récupérer ou définir les colonnes exactes
colonnes_modele = obtenir_colonnes_model(modele)  # Cette fonction doit renvoyer les bonnes colonnes du modèle

# Fonction de prétraitement avec get_dummies
def pretraiter_donnees(donnees_brutes):
    # Encodage des variables catégorielles avec get_dummies
    donnees_encodees = pd.get_dummies(donnees_brutes)
    
    # Ajouter les colonnes manquantes avec 0 pour assurer que le nombre de colonnes est le même
    for colonne in colonnes_modele:
        if colonne not in donnees_encodees.columns:
            donnees_encodees[colonne] = 0
    
    # Garder uniquement les colonnes nécessaires dans le bon ordre
    donnees_encodees = donnees_encodees[colonnes_modele]
    
    return donnees_encodees

# Interface utilisateur
st.title("Prédiction de Revenu Annuel")

st.markdown("""
Cette application prédit si le revenu annuel d'une personne dépasse **50 000$**  
en fonction de ses caractéristiques socio-démographiques.
""")

with st.form("formulaire_entree"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Âge", min_value=18, max_value=100, value=30)
        classe_pro = st.selectbox("Classe professionnelle", 
                                  ['Privé', 'Auto-emploi', 'Gouvernement', 'Autre'])
        education = st.selectbox("Niveau d'éducation", 
                                 ['Licence', 'Bac', 'Master', 'Doctorat'])
    
    with col2:
        statut_matrimonial = st.selectbox("Statut matrimonial", 
                                           ['Marié(e)', 'Divorcé(e)', 'Célibataire'])
        profession = st.selectbox("Profession", 
                                  ['Technique', 'Ventes', 'Administratif', 'Autre'])
        situation_familiale = st.selectbox("Situation familiale", 
                                           ['Conjoint', 'Conjointe', 'Sans conjoint'])
    
    with col3:
        origine = st.selectbox("Origine ethnique", ['Blanc', 'Noir', 'Asiatique'])
        sexe = st.radio("Genre", ['Homme', 'Femme'])
        heures_semaine = st.slider("Heures travaillées par semaine", 1, 80, 40)
    
    soumettre = st.form_submit_button("Effectuer la prédiction")

if soumettre and modele is not None:
    # Créer un DataFrame avec les données saisies
    donnees_entree = pd.DataFrame([{
        'age': age,
        'workclass': classe_pro,
        'education': education,
        'marital.status': statut_matrimonial,
        'occupation': profession,
        'relationship': situation_familiale,
        'race': origine,
        'sex': sexe,
        'hours.per.week': heures_semaine
    }])
    
    try:
        # Prétraitement et prédiction
        donnees_traitees = pretraiter_donnees(donnees_entree)
        
        # Vérifier si le nombre de colonnes correspond à celui attendu par le modèle
        if donnees_traitees.shape[1] != len(colonnes_modele):
            st.error(f"Le nombre de colonnes après prétraitement est incorrect ({donnees_traitees.shape[1]} au lieu de {len(colonnes_modele)}).")
        else:
            prediction = modele.predict(donnees_traitees)[0]
            probabilite = modele.predict_proba(donnees_traitees)[0][1]
            
            # Affichage des résultats
            if prediction == 1:
                st.success(f"Prédiction : Revenu > 50 000$ (Probabilité : {probabilite:.1%})")
                st.balloons()
            else:
                st.warning(f"Prédiction : Revenu ≤ 50 000$ (Probabilité : {probabilite:.1%})")
                    
    except Exception as e:
        st.error("Une erreur est survenue lors de la prédiction")
        st.error(f"Détails : {str(e)}")
else:
    if modele is None:
        st.warning("Le modèle n'a pas été chargé correctement. Vérifiez le fichier adulte.joblib.")
