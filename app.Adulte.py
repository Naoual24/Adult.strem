import streamlit as st
import joblib
import pandas as pd

# Charger le modèle et les colonnes attendues
@st.cache_resource
def charger_modele_et_colonnes():
    try:
        modele = joblib.load("adulte.joblib")
        colonnes = joblib.load("adulte.joblib")  # Fichier séparé pour les colonnes
        return modele, colonnes
    except Exception as e:
        st.error(f"Erreur lors du chargement : {e}")
        return None, None

modele, colonnes_modele = charger_modele_et_colonnes()

# Fonction de prétraitement
def pretraiter_donnees(donnees_brutes):
    # Encodage des variables catégorielles
    donnees_encodees = pd.get_dummies(donnees_brutes)
    
    # Ajouter les colonnes manquantes avec 0
    for colonne in colonnes_modele:
        if colonne not in donnees_encodees.columns:
            donnees_encodees[colonne] = 0
    
    # Garder uniquement les colonnes nécessaires dans le bon ordre
    return donnees_encodees[colonnes_modele]

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
        st.error(f"Détails de l'erreur : {str(e)}")
        st.info("Veuillez vérifier que toutes les données sont correctement renseignées.")

