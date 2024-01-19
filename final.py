import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", None)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

data_0 = pd.read_csv("lower_extremity_amputation.csv")
data_1 = data_0.iloc[:, 1:22] #sadrzi elemente od kolona B do kolone V ukljucujuci i njih

y_1 = data_0.iloc[:, 22]
y_2 = data_0.iloc[:, 23]
y_1 = y_1.to_frame()
y_2 = y_2.to_frame()

value_mapping = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}

y_1['NIVO OSPOSOBLJENOSTI ZA HOD'].replace(value_mapping, inplace=True)
y_2['K LEVEL'].replace(value_mapping, inplace=True) #objedinjavanje klasa tako da 0,1 pripadaju klasi 0, 2 klasi 1, a 3,4 klasi 2 - izlazi kompletni

#objedinjavanje 4 kolona koje nose informaciju o nivou pokretnosti pacijenata
class_columns = data_0.iloc[:, 29:33]
class_column_indices = np.argmax(class_columns.values, axis=1)
class_values = class_column_indices + 1
data_0['Pokretnost'] = class_values

data_1['Pokretnost'] = data_0['Pokretnost'] #dodavanje na df sa preostalim ulazima - ulazi kompletni

#zamena nedostajucih podataka sa srednjom u slucaju numerickih, i najcescom klasom u slucaju kategoricnih vrednosti
data_1['BDI'].fillna(data_1['BDI'].mean(), inplace=True)
data_1.at[11, 'Fantomski bol'] = np.nan
data_1['Fantomski bol'].fillna(data_1['Fantomski bol'].mode().iloc[0], inplace=True)
data_1['Fantomski bol'] = data_1['Fantomski bol'].astype(int)
data_1['pušac (0 nije, 1 jeste, 2 nije duže od 6 meseci)'].fillna(data_1['pušac (0 nije, 1 jeste, 2 nije duže od 6 meseci)'].mode().iloc[0], inplace=True)
data_1['TUG'].fillna(data_1['TUG'].mean(), inplace=True)
data_1['2-minute walk test'].fillna(data_1['2-minute walk test'].mean(), inplace=True)

categorical_columns = ['POL', 'Uzrok amputacije', 'Nivo amputacije', 'Dijabetes', 'Fantomski bol', 'Kontraktura', 'RE ekstenzori kuka CELE OCENE',
                       'IE plantarni fleksori CELE OCENE', 'IE ekstenzori kuka CELE OCENE', 'Balans', 'pušac (0 nije, 1 jeste, 2 nije duže od 6 meseci)']
label_encoder = LabelEncoder()

for col in categorical_columns:
    data_1[col] = label_encoder.fit_transform(data_1[col])

X = data_1.values
y1 = y_1.values
y2 = y_2.values

#pronalazenje najboljih obelezja i ujedno ispisivanje istih
selector = SelectKBest(score_func=chi2, k=5)
X1 = selector.fit_transform(X, y1)
selected_feature_indices = selector.get_support(indices=True)
selected_feature_names = data_1.columns[selected_feature_indices]
print(list(selected_feature_names))

X2 = selector.fit_transform(X, y2)
selected_feature_indices = selector.get_support(indices=True)
selected_feature_names = data_1.columns[selected_feature_indices]
print(list(selected_feature_names))

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.3, stratify=y1)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3, stratify=y2)

models = [SVC(), LogisticRegression(), KNeighborsClassifier(), DecisionTreeClassifier()]
X = [[X_train1, X_test1],[X_train2, X_test2]]
y = [[y_train1, y_test1],[y_train2, y_test2]]

for i in range(2):
    if (i == 0):
        print("Predikcija - NIVO OSPOSOBLJENOSTI ZA HOD")
    else:
        print("Predikcija - K LEVEL")

    for model in models:
        name = model.__class__.__name__
        if name == "LogisticRegression":
            hiperparametri = {
                'penalty': ['l1', 'l2'],
                'C': [0.1, 1, 5, 10, 25, 50, 100],
                'solver': ['liblinear', 'saga']
            }
        if name == "KNeighborsClassifier":
            hiperparametri = {
                'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        if name == "DecisionTreeClassifier":
            hiperparametri = {
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 5, 10],
                'min_samples_split': [2, 5, 10]
            }
        if name == "SVC":
            hiperparametri = {
                'C': [0.01, 0.1, 1, 10, 100],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'degree': [3, 5, 7],
                'gamma': ['scale', 'auto']
            }

        grid = GridSearchCV(model, hiperparametri, cv=10)
        grid.fit(X[i][0], y[i][0])

        model.set_params(**grid.best_params_)
        model.fit(X[i][0], y[i][0])
        y_predict = model.predict(X[i][1])

        print("Rezultati za algoritam " + name + "\n")
        print("Accuracy score: ", "%.2f" % (accuracy_score(y[i][1], y_predict) * 100), "%")
        print("Precision score: ", "%.2f" % (precision_score(y[i][1], y_predict, average='weighted') * 100), "%") #ne radi ako ne stavim weighted
        print("Recall score: ", "%.2f" % (recall_score(y[i][1], y_predict, average='weighted') * 100), "%")
        print("F1 score: ", "%.2f" % (f1_score(y[i][1], y_predict, average='weighted') * 100), "%")
        print("Confusion matrix: \n", confusion_matrix(y[i][1], y_predict))
