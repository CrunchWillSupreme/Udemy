# SVR

import os
os.chdir(r'C:\Users\willh\Udemy\Machine Learning A-Z\P14-Machine-Learning-AZ-Template-Folder\Part 2 - Regression\Section 7 - Support Vector Regression (SVR)')

# Multiple Linear Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split --deprecated
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
y=y.reshape(-1,1)
X = sc_X.fit_transform(X) 
y = sc_y.fit_transform(y)

# Fitting the SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf') #rbf is gaussian
regressor.fit(X, y)

# Predicting a new result with Polynomial Regression
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

# Visualizing the SVRresults (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth of Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# Visualizing the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth of Bluff (SVR Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
