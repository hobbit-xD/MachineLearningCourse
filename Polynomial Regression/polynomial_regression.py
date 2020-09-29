# Polynomial regression

# Importing the libraries
import numpy as np
import mathplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
froms sklearn.preprocessing import PolynomialFeatures

# Importing the dataset
dataset = pd.read_csv("data.csv")
X = dataset.iloc[:,1:2].values #: --> di tutte le linee prende tutte le colonne, tranne l' ultima colonna
Y = dataset.iloc[:,2].values # prende l'ultima colonna di tutte le linee

'''
# Splitting dataset into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

# Feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

sc_Y = StandardScaler()
Y_train = sc_Y.fit_transform(Y_train)
'''

# Fitting Linear Regression to the dataset
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

# Fitting Polynomial Regression to the dataset
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,Y)

# Visualizing the Linear Regression results
plt.scatter(X,Y, color = "red")
plt.plot(X, lin_reg.predict(X), color = "blue")
plt.title("Truth or Bluff (LinearRegression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualizing the Polynomial Regression results
X_grid = np.arange(min(X),max(X), 0.1)
X_grid = X_grid.reshape((len(x_grid),1))
plt.scatter(X,Y, color = "red")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = "blue")
plt.title("Truth or Bluff (PolynomialRegression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Predicting a new result with linear regression
lin_reg.predict(6.5)

# Predicting a new result with polynomial regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))

