#Exercice 1


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay



# 1. Charger le jeu de données
url = "spam.csv"  # Remplacez par le chemin correct de votre fichier CSV


# Nettoyage et Rename des colonnes en label et message
data = pd.read_csv(url, encoding='latin-1', header=None, names=['label', 'message'], usecols=[0, 1])

# Vérification du contenu des premières lignes pour s'assurer qu'il est bien chargé
print(data.head())

# 2. Prétraitement des données

# Convertir le texte en minuscules et enlever les caractères non-alphabétiques
data['message'] = data['message'].str.replace(r'\W+', ' ', regex=True).str.lower()

# Convertir les étiquettes "ham" et "spam" en 0 et 1
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# 3. Vérifier si des NaN existent dans la colonne 'label' et les supprimer
if data['label'].isnull().any():
    print("Il y a des valeurs NaN dans la colonne 'label'. Nous allons les supprimer.")
    data = data.dropna(subset=['label'])

# Vérification après nettoyage
print(f"Après nettoyage, nombre de messages : {len(data)}")
print(data.head())  # Afficher les 5 premières lignes pour vérifier

# 4. Vectorisation des messages
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['message'])
y = data['label']

# 5. Division des données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 6. Entraînement du modèle Naïve Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# 7. Prédictions et évaluation du modèle
y_pred = nb_model.predict(X_test)

# 8. Affichage des performances du modèle
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))




#############################################################################################



# Exercice 2 : Classification avec Complement Naive Bayes

# Formation du modèle Complement Naive Bayes
cnb_model = ComplementNB()
cnb_model.fit(X_train, y_train)

# 7. Prédictions et évaluation du modèle
y_pred_cnb = cnb_model.predict(X_test)

# 8. Affichage des performances du modèle
print("Accuracy (Complement Naive Bayes):", accuracy_score(y_test, y_pred_cnb))
print("Classification Report (Complement Naive Bayes):\n", classification_report(y_test, y_pred_cnb))

# 9. Matrice de confusion pour mieux analyser les erreurs
cm = confusion_matrix(y_test, y_pred_cnb)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
disp.plot()



#############################################################################################


# Exercice 3 : Régression Logistique

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# 1. Charger le jeu de données
url = "spam.csv"  # Remplacez par le chemin correct de votre fichier CSV

# Nettoyage et Rename des colonnes en label et message
data = pd.read_csv(url, encoding='latin-1', header=None, names=['label', 'message'], usecols=[0, 1])

# Vérification du contenu des premières lignes pour s'assurer qu'il est bien chargé
print(data.head())

# 2. Prétraitement des données

# Convertir le texte en minuscules et enlever les caractères non-alphabétiques
data['message'] = data['message'].str.replace(r'\W+', ' ', regex=True).str.lower()

# Convertir les étiquettes "ham" et "spam" en 0 et 1
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# 3. Vérifier si des NaN existent dans la colonne 'label' et les supprimer
if data['label'].isnull().any():
    print("Il y a des valeurs NaN dans la colonne 'label'. Nous allons les supprimer.")
    data = data.dropna(subset=['label'])

# Vérification après nettoyage
print(f"Après nettoyage, nombre de messages : {len(data)}")
print(data.head())  # Afficher les 5 premières lignes pour vérifier

# 4. Vectorisation des messages
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['message'])
y = data['label']

# 5. Division des données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 6. Entraînement du modèle de régression logistique
log_reg_model = LogisticRegression(max_iter=1000)  # Ajustez max_iter si nécessaire
log_reg_model.fit(X_train, y_train)

# 7. Prédictions et évaluation du modèle
y_pred_lr = log_reg_model.predict(X_test)

# 8. Affichage des performances du modèle
print("Accuracy (Logistic Regression):", accuracy_score(y_test, y_pred_lr))
print("Classification Report (Logistic Regression):\n", classification_report(y_test, y_pred_lr))

# 9. Matrice de confusion pour mieux analyser les erreurs
cm_lr = confusion_matrix(y_test, y_pred_lr)
disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=["Ham", "Spam"])
disp_lr.plot()



#############################################################################################



# Exercice 4 : Comparaison des Performances

import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Créer une table de comparaison des performances

# Récupérer les métriques pour chaque modèle
performance_data = {
    'Model': ['Naive Bayes', 'Complement Naive Bayes', 'Logistic Regression'],
    'Precision': [
        classification_report(y_test, y_pred, output_dict=True),
        classification_report(y_test, y_pred_cnb, output_dict=True),
        classification_report(y_test, y_pred_lr, output_dict=True),
    ],
    'Recall': [
        classification_report(y_test, y_pred, output_dict=True),
        classification_report(y_test, y_pred_cnb, output_dict=True),
        classification_report(y_test, y_pred_lr, output_dict=True),
    ],
    'F1-score': [
        classification_report(y_test, y_pred, output_dict=True),  
        classification_report(y_test, y_pred_cnb, output_dict=True),  
        classification_report(y_test, y_pred_lr, output_dict=True),
    ]
}

# Convertir en DataFrame pour une meilleure lisibilité
performance_df = pd.DataFrame(performance_data)
print(performance_df)


