# Simple linear regression

# Importing the libraries
import numpy as np
import mathplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv("data.csv")
X = dataset.iloc[:,:-1].values #: --> di tutte le linee prende tutte le colonne, tranne l' ultima colonna
Y = dataset.iloc[:,1].values # prende l'ultima colonna di tutte le linee

# Splitting dataset into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 1/3, random_state = 0)

'''
# Feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

sc_Y = StandardScaler()
Y_train = sc_Y.fit_transform(Y_train)
'''

# Fitting Simple Linear regression to the training set
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, Y_train, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years")
plt.ylabel("Salary")
plt.show()

# Visualising the Test set results
plt.scatter(X_test, Y_test, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = "blue") #NON serve cambiare tanto si ottiene sempre la stessa linea
plt.title("Salary vs Experience (Test Set)")
plt.xlabel("Years")
plt.ylabel("Salary")
plt.show()

