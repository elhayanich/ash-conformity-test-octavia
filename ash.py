import json
import pandas as pd
import csv
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Fonction ash qui vérifie si la réponse est correcte
def ash(ref, A, B, C, rep):
    if ref == A and rep == 1:
        return True
    elif ref == B and rep == 2:
        return True
    elif ref == C and rep == 3:
        return True
    return False

# Test de la fonction ash
test_data_true = [10, 5, 10, 20, 2]
print(ash(*test_data_true))  # Devrait retourner True

test_data_false = [10, 5, 10, 20, 3]
print(ash(*test_data_false))  # Devrait retourner False

test_data_false_2 = [3, 5, 3, 4, 2]
print(ash(*test_data_false_2))  # Devrait retourner False

# Lecture des données CSV
with open('ash_train_valid_40.csv') as csvfile:
    data = list(csv.reader(csvfile))

# Suppression de l'en-tête
data = data[1:]

# Conversion des valeurs 'true'/'false' et numériques
data_temp = []
for l in data:
    l_temp = []
    for case in l:
        case = case.lower().strip()  # Gestion des valeurs TRUE, TRUE, et espaces
        if case == "true":
            l_temp.append(1)
        elif case == "false":
            l_temp.append(0)
        else:
            try:
                l_temp.append(int(case))  # Conversion des autres valeurs en entier
            except ValueError:
                print(f"Erreur de conversion dans la valeur: {case}")  # Pour déboguer si des erreurs surviennent
                continue
    data_temp.append(l_temp)
data = data_temp

# Affichage des données après conversion
print(data)

# Vérification des lignes avec la fonction ash
compteur = 1
for l in data:
    compteur += 1
    if not ash(*l[:5]):  # Vérification de la validité
        print("PROBLEME !!!")
        print("ligne:", compteur)
        print(l)

### Machine Learning avec un arbre de décision

# Lecture des données pour l'entraînement du modèle
df = pd.read_csv("ash_train_valid_35.csv")

# Mapping des booléens en entiers
df["valid"] = df["valid"].map({True: 1, False: 0})

# Sélection des caractéristiques et de la cible
features = ['ref', 'A', 'B', 'C', 'rep']
inputs = df[features]
outputs = df['valid']

# Création et entraînement du modèle d'arbre de décision
dtree = DecisionTreeClassifier()
dtree = dtree.fit(inputs, outputs)

# Visualisation de l'arbre de décision

tree.plot_tree(dtree, feature_names=features)
plt.savefig('decision_tree.png')
plt.show()

# Exemple de prédiction
print(dtree.predict([[40, 10, 7, 1]]))  # Exemple de prédiction
print(dtree.predict([[40, 10, 6, 1]]))  # Exemple de prédiction

