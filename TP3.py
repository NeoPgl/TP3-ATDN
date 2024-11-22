#Exercice 1


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay


print("======================================Exercice 1 : ===================================================\n")




# 1. Charger le jeu de données
url = "spam.csv"  


# Nettoyage et Rename des colonnes en label et message
data = pd.read_csv(url, encoding='latin-1', header=None, names=['label', 'message'], usecols=[0, 1])


# 2. Prétraitement des données

# Convertir le texte en minuscules et enlever les caractères non-alphabétiques
data['message'] = data['message'].str.replace(r'\W+', ' ', regex=True).str.lower()

# Convertir les étiquettes "ham" et "spam" en 0 et 1
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Supprimer les valeurs NaN existant dans la colonne 'label'
if data['label'].isnull().any():
    data = data.dropna(subset=['label'])

# Vérification après nettoyage
print(f"Après nettoyage, nombre de messages : {len(data)}")
print(data.head())  # Afficher les 5 premières lignes pour vérifier

# Vectorisation des messages
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['message'])
y = data['label']


#3. Division des données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# 4. Entraînement du modèle Naïve Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# 5. Prédictions et évaluation du modèle
y_pred = nb_model.predict(X_test)

# Affichage des performances du modèle en utilisant les métriques
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))




#############################################################################################



# Exercice 2 : Classification avec Complement Naive Bayes

import matplotlib.pyplot as plt



# Le jeu de données est chargé plus haut dans l'exercice précédent, donc on ne le recharge pas ici.
# Pas besoin de nettoyer et vectoriser les données car c'est également déjà fait dans l'exo 1.


print("======================================Exercice 2 : ===================================================\n")


# 3. Formation du modèle Complement Naive Bayes
cnb_model = ComplementNB()
cnb_model.fit(X_train, y_train)

# Prédictions et évaluation du modèle
y_pred_cnb = cnb_model.predict(X_test)

# 4. Affichage des performances du modèle en utilisant les métriques
print("Accuracy (Complement Naive Bayes):", accuracy_score(y_test, y_pred_cnb))
print("Classification Report (Complement Naive Bayes):\n", classification_report(y_test, y_pred_cnb))


# 5. Matrice de confusion
cm = confusion_matrix(y_test, y_pred_cnb)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
disp.plot()
plt.show()



#############################################################################################


# Exercice 3 : Régression Logistique

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay


print("======================================Exercice 3 : ===================================================\n")


# 1. Charger le jeu de données
url = "spam.csv"  


# Nettoyage et Rename des colonnes en label et message
data = pd.read_csv(url, encoding='latin-1', header=None, names=['label', 'message'], usecols=[0, 1])


# 2. Prétraitement des données

# Convertir le texte en minuscules et enlever les caractères non-alphabétiques
data['message'] = data['message'].str.replace(r'\W+', ' ', regex=True).str.lower()

# Convertir les étiquettes "ham" et "spam" en 0 et 1
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Supprimer les valeurs NaN existant dans la colonne 'label'
if data['label'].isnull().any():
    data = data.dropna(subset=['label'])

# Vérification après nettoyage
print(f"Après nettoyage, nombre de messages : {len(data)}")
print(data.head())  # Afficher les 5 premières lignes pour vérifier

# Vectorisation des messages
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['message'])
y = data['label']


#3. Division des données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Entraînement du modèle de régression logistique
log_reg_model = LogisticRegression(max_iter=1000)  # Ajustez max_iter si nécessaire
log_reg_model.fit(X_train, y_train)

# 5. Prédictions et évaluation du modèle
y_pred_lr = log_reg_model.predict(X_test)

# Affichage des performances du modèle en utilisant les métriques
print("Accuracy (Logistic Regression):", accuracy_score(y_test, y_pred_lr))
print("Classification Report (Logistic Regression):\n", classification_report(y_test, y_pred_lr))




#############################################################################################



# Exercice 4 : Comparaison des Performances

import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MaxAbsScaler



print("======================================Exercice 4 : ===================================================\n")


# 1. Table de comparaison des performances

performance_data = {
    'Model': ['Naive Bayes', 'Complement Naive Bayes', 'Régression Logistique'],
    'Précision': [
        classification_report(y_test, y_pred, output_dict=True),
        classification_report(y_test, y_pred_cnb, output_dict=True),
        classification_report(y_test, y_pred_lr, output_dict=True),
    ],
    'Rappel': [
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




print("======================================Discussions : ===================================================\n")


import time


# Liste des modèles
models = {
    "Multinomial Naive Bayes": MultinomialNB(),
    "Complement Naive Bayes": ComplementNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

# Comparaison des temps d'entraînement
training_times = {}

for model_name, model in models.items():
    start_time = time.time()  # Démarrer le chronomètre
    model.fit(X_train, y_train)  # Entraîner le modèle
    end_time = time.time()  # Arrêter le chronomètre
    training_times[model_name] = end_time - start_time  # Calculer la durée

# Affichage des résultats
print("Temps d'entraînement des modèles :")
for model_name, training_time in training_times.items():
    print(f"{model_name}: {training_time:.4f} secondes")