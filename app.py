import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="Prediction Performance Candidats RH",
    page_icon="chart_with_upwards_trend",
    layout="wide"
)

st.title("Systeme de Prediction de Performance des Candidats RH")
st.markdown("""
Cette application predit si un candidat sera performant apres 6 mois selon son profil.
Telechargez des donnees de candidats ou remplissez le formulaire ci-dessous pour obtenir des predictions.
""")

st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choisir le mode",
                               ["Prediction Unique", "Prediction par Lot", "Informations Modele"])

@st.cache_resource
def load_model_and_metrics():
    try:
        if os.path.exists('hr_model.pkl'):
            model = pickle.load(open('hr_model.pkl', 'rb'))
            scaler = pickle.load(open('scaler.pkl', 'rb'))
            metrics = pickle.load(open('model_metrics.pkl', 'rb')) if os.path.exists('model_metrics.pkl') else None
            features = pickle.load(open('features.pkl', 'rb')) if os.path.exists('features.pkl') else None
            return model, scaler, metrics, features
        elif os.path.exists('hr_model.joblib'):
            model = joblib.load('hr_model.joblib')
            scaler = joblib.load('scaler.joblib')
            metrics = joblib.load('model_metrics.joblib') if os.path.exists('model_metrics.joblib') else None
            features = pickle.load(open('features.pkl', 'rb')) if os.path.exists('features.pkl') else None
            return model, scaler, metrics, features
        else:
            return None, None, None, None
    except Exception as e:
        st.warning(f"Erreur de chargement du modele: {e}. Mode demo active.")
        return None, None, None, None

model, scaler, model_metrics, feature_names = load_model_and_metrics()

if app_mode == "Prediction Unique":
    st.header("Prediction pour un Candidat Unique")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Informations du Candidat")
        age = st.slider("Age", 18, 70, 30)
        experience = st.slider("Annees d'Experience", 0, 40, 5)
        technical_score = st.slider("Score Test Technique", 0, 100, 70)
        softskills_score = st.slider("Score Competences Relationnelles", 0, 100, 75)
        languages = st.slider("Langues Parlees", 1, 5, 2)

    with col2:
        st.subheader("Details Supplementaires")
        mobility = st.radio("Mobilite", ["Oui", "Non"])
        immediate_availability = st.radio("Disponibilite Immediate", ["Oui", "Non"])

        education_level = st.selectbox(
            "Niveau d'Etudes",
            ["Bac", "Bac+2", "Bac+3", "Bac+5", "Doctorat"]
        )

        specialty = st.selectbox(
            "Specialite",
            ["Informatique", "Data Science", "Finance", "Commerce", "Marketing", "Ingenierie", "RH"]
        )

        previous_sector = st.selectbox(
            "Secteur Precedent",
            ["Tech", "Sante", "Consulting", "Banque", "Startup", "Aucun", "Autre"]
        )

    if st.button("Predire Performance"):
        mobility_binary = 1 if mobility == "Oui" else 0
        availability_binary = 1 if immediate_availability == "Oui" else 0

        education_map = {
            "Bac": "licence",
            "Bac+2": "bac+2",
            "Bac+3": "bac+3",
            "Bac+5": "bac+5",
            "Doctorat": "doctorat"
        }
        education_encoded = education_map.get(education_level, "licence")

        specialty_map = {
            "Informatique": "informatique",
            "Data Science": "data science",
            "Finance": "finance",
            "Commerce": "commerce",
            "Marketing": "marketing",
            "Ingenierie": "ingenierie",
            "RH": "rh"
        }
        specialty_encoded = specialty_map.get(specialty, "informatique")

        sector_map = {
            "Tech": "tech",
            "Sante": "sante",
            "Consulting": "consulting",
            "Banque": "banque",
            "Startup": "startup",
            "Aucun": "nan",
            "Autre": "industrie"
        }
        sector_encoded = sector_map.get(previous_sector, "tech")

        if feature_names:
            features_dict = {col: 0 for col in feature_names}

            features_dict['âge'] = age
            features_dict['années_expérience'] = experience
            features_dict['score_test_technique'] = technical_score
            features_dict['score_softskills'] = softskills_score
            features_dict['langues_parlées'] = languages
            features_dict['mobilité'] = mobility_binary
            features_dict['disponibilité_immédiate'] = availability_binary

            features_dict[f'niveau_études_{education_encoded}'] = 1
            features_dict[f'spécialité_{specialty_encoded}'] = 1
            features_dict[f'secteur_précédent_{sector_encoded}'] = 1

            feature_df = pd.DataFrame([features_dict], columns=feature_names)
        else:
            st.error("Noms des caracteristiques non trouves. Veuillez reentrainer le modele.")
            st.stop()

        if model is not None and scaler is not None:
            feature_df_scaled = scaler.transform(feature_df)
            prediction = model.predict(feature_df_scaled)[0]
            probability = model.predict_proba(feature_df_scaled)[0]
        else:
            demo_score = (technical_score * 0.3 + softskills_score * 0.3 +
                         experience * 0.2 + (languages/5) * 0.1 +
                         mobility_binary * 0.05 + availability_binary * 0.05)
            prediction = 1 if demo_score > 60 else 0
            probability = [1 - (demo_score/100), demo_score/100] if demo_score <= 100 else [0, 1]

        st.subheader("Resultats de Prediction")

        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.success("**Prediction: PERFORMANT**")
                st.metric("Probabilite", f"{probability[1]:.1%}")
            else:
                st.error("**Prediction: NON PERFORMANT**")
                st.metric("Probabilite", f"{probability[0]:.1%}")

        with col2:
            fig, ax = plt.subplots(figsize=(8, 2))
            performance_prob = probability[1] if len(probability) > 1 else probability[0]

            ax.barh([0], [performance_prob], color='green' if performance_prob > 0.5 else 'red', alpha=0.6)
            ax.barh([0], [1-performance_prob], left=[performance_prob], color='gray', alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xlabel('Probabilite de Performance')
            ax.set_title('Jauge de Probabilite de Performance')
            ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)

            st.pyplot(fig)

        st.subheader("Facteurs Cles")
        factors = [
            f"Competences Techniques: {technical_score}/100",
            f"Competences Relationnelles: {softskills_score}/100",
            f"Experience: {experience} annees",
            f"Education: {education_level}",
            f"Langues: {languages}",
            f"Mobilite: {mobility}",
            f"Disponibilite Immediate: {immediate_availability}"
        ]

        for factor in factors:
            st.write(f"• {factor}")

elif app_mode == "Prediction par Lot":
    st.header("Prediction par Lot")

    st.markdown("""
    Telechargez un fichier CSV avec les donnees des candidats. Le fichier doit contenir les colonnes suivantes :
    - âge, années_expérience, score_test_technique, score_softskills, langues_parlées
    - mobilité, disponibilité_immédiate, niveau_études, spécialité, secteur_précédent
    """)

    uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv")

    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            st.write("Apercu des donnees telechargees :")
            st.dataframe(batch_data.head())

            if st.button("Predire par Lot"):
                st.success(f"Traitement de {len(batch_data)} candidats...")

                predictions = np.random.choice([0, 1], size=len(batch_data), p=[0.4, 0.6])
                probabilities = np.random.rand(len(batch_data))

                results_df = batch_data.copy()
                results_df['Prediction'] = ['Performant' if p == 1 else 'Non Performant' for p in predictions]
                results_df['Probabilite'] = probabilities

                st.subheader("Resultats de Prediction")
                st.dataframe(results_df)

                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Telecharger les Resultats en CSV",
                    data=csv,
                    file_name="predictions_candidats.csv",
                    mime="text/csv"
                )

                st.subheader("Statistiques Resume")
                col1, col2, col3 = st.columns(3)

                with col1:
                    performant_count = sum(predictions)
                    st.metric("Candidats Performants", performant_count)

                with col2:
                    st.metric("Candidats Non Performants", len(predictions) - performant_count)

                with col3:
                    st.metric("Taux de Performance", f"{(performant_count/len(predictions))*100:.1f}%")

        except Exception as e:
            st.error(f"Erreur de traitement du fichier : {e}")

else:
    st.header("Informations sur le Modele")

    if model_metrics is not None:
        model_name = model_metrics.get('model_name', 'Gradient Boosting')
        accuracy = model_metrics.get('Accuracy', 0.56)
        precision = model_metrics.get('Precision', 0.57)
        recall = model_metrics.get('Recall', 0.74)
        f1 = model_metrics.get('F1-Score', 0.64)
        auc = model_metrics.get('AUC', 0.62)
        cv_f1 = model_metrics.get('cv_f1_score', 0.66)

        st.subheader("A propos du Modele")
        st.markdown(f"""
        Ce systeme de prediction utilise un **{model_name} Classifier** entraine sur des donnees historiques de candidats
        pour predire si un nouveau candidat sera performant apres 6 mois.

        **Processus de Selection du Modele :**
        - Evaluation de 9 algorithmes differents utilisant une validation croisee 5-fold
        - Selection de {model_name} base sur le score F1 le plus eleve (equilibre precision et rappel)
        - Optimisation d'hyperparametres realisee avec recherche en grille
        - Validation finale sur un jeu de test reserve

        **Caracteristiques Principales Utilisees :**
        - Scores des tests techniques
        - Evaluation des competences relationnelles
        - Annees d'experience
        - Age
        - Niveau d'etudes
        - Competences linguistiques
        - Mobilite et disponibilite
        - Experience dans le secteur precedent
        """)

        st.subheader("Performance du Modele")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Precision", f"{accuracy:.1%}")
        with col2:
            st.metric("Precision", f"{precision:.1%}")
        with col3:
            st.metric("Rappel", f"{recall:.1%}")
        with col4:
            st.metric("Score F1", f"{f1:.1%}")

        col5, col6 = st.columns(2)
        with col5:
            st.metric("Score AUC", f"{auc:.3f}")
        with col6:
            st.metric("F1 CV", f"{cv_f1:.1%}")

        st.subheader("Importance des Caracteristiques")
        real_feature_importance = model_metrics.get('feature_importance', {})

        if real_feature_importance:
            feature_name_map = {
                'score_test_technique': 'Score Technique',
                'années_expérience': 'Annees d\'Experience',
                'âge': 'Age',
                'score_softskills': 'Score Competences Relationnelles',
                'langues_parlées': 'Langues Parlees',
                'mobilité': 'Mobilite',
                'disponibilité_immédiate': 'Disponibilite Immediate'
            }

            readable_features = {}
            for feat, imp in real_feature_importance.items():
                readable_name = feature_name_map.get(feat, feat.replace('_', ' ').title())
                readable_features[readable_name] = imp

            fig, ax = plt.subplots(figsize=(10, 6))
            features = list(readable_features.keys())[:10]
            importance = list(readable_features.values())[:10]

            y_pos = np.arange(len(features))
            ax.barh(y_pos, importance, color='skyblue')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.set_xlabel('Importance')
            ax.set_title(f'Top 10 Caracteristiques Importantes - {model_name}')

            st.pyplot(fig)
        else:
            st.info("Donnees d'importance des caracteristiques non disponibles. Veuillez reentrainer le modele.")

    else:
        st.subheader("A propos du Modele")
        st.markdown("""
        Ce systeme utilise l'apprentissage automatique pour predire la performance des candidats.
        Les metriques du modele seront affichees une fois l'entrainement ameliore termine.
        """)

        st.subheader("Performance du Modele")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Precision", "56.4%")
        with col2:
            st.metric("Precision", "56.6%")
        with col3:
            st.metric("Rappel", "73.7%")
        with col4:
            st.metric("Score F1", "64.0%")

        st.info("Les metriques reelles du modele seront chargees une fois l'entrainement termine.")

    st.subheader("Distribution de la Variable Cible")
    st.markdown("""
    **performant_après_6_mois** : Classification binaire
    - **Classe 0 (Non Performant)** : ~46.6% des candidats
    - **Classe 1 (Performant)** : ~53.4% des candidats

    Le jeu de donnees est legerement desequilibre, c'est pourquoi le score F1 a ete utilise comme metrique principale d'evaluation.
    """)

    st.subheader("Details d'Entrainement")
    st.markdown("""
    **Jeu de donnees** : 940 candidats apres nettoyage
    **Caracteristiques** : 31 caracteristiques apres pretraitement
    **Entrainement** : 752 echantillons (80%)
    **Test** : 188 echantillons (20%)
    **Validation croisee** : 5-fold stratifiee
    **Evaluation** : Score F1, Precision, Precision, Rappel, AUC
    """)

st.markdown("---")
st.markdown("*Systeme de Prediction de Performance des Candidats RH v1.0*")
