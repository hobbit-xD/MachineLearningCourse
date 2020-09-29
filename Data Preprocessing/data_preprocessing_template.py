# Data Preprocessing

# Importing the libraries
import numpy as np
import mathplotlib.pyplot as plt
import pandas as pd
from sklearn.prepocessing import Imputer
from sklearn.prepocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.prepocessing import StandardScaler

# Importing the dataset
dataset = pd.read_csv("data.csv")
X = dataset.iloc[:,:-1].values #: --> di tutte le linee prende tutte le colonne, tranne l' ultima colonna
Y = dataset.iloc[:,3].values # prende l'ultima colonna di tutte le linee

# Missing data
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = "0")
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3]) # sostituisce i valori mancanti con la media nella matrice iniziale

# Encoding categorical data
labelEncoder_X = LabelEncoder()
X[:,0] = labelEncoder_X.fit_transform(X[:,0])
oneHotEncoder = OneHotEncoder(categorical_features = [0]) #Dummy encoding
X = oneHotEncoder.fit_transform(X).toarray()

labelEncoder_Y = LabelEncoder()
Y = labelEncoder_Y.fit_transform(Y)

# Splitting dataset into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

# Feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

sc_Y = StandardScaler()
Y_train = sc_Y.fit_transform(Y_train)
