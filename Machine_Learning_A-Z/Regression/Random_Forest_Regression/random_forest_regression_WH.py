# Random Forest Regression
import os
os.chdir(r'C:\Users\willh\Udemy\Machine Learning A-Z\P14-Machine-Learning-AZ-Template-Folder\Part 2 - Regression\Section 9 - Random Forest Regression')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# REMOVING THIS STEP FOR TEMPLATE
## Taking care of missing data
#from sklearn.impute import SimpleImputer
#imputer = SimpleImputer(strategy = 'mean')
#imputer = imputer.fit(X[:, 1:3])
#X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X = LabelEncoder()
#X[:, -1] = labelencoder_X.fit_transform(X[:, -1])
#onehotencoder = OneHotEncoder(categorical_features = [-1])
#X = onehotencoder.fit_transform(X).toarray()

#labelencoder_y = LabelEncoder()
#y = labelencoder_y.fit_transform(y)

# Avoiding the Dummy Variable Trap - take away one of the dummy variables
#X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split --deprecated
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) 
X_test = sc_X.transform(X_test)"""

## Fitting Linear Regression to the Training set
#from sklearn.linear_model import LinearRegression
#lin_reg = LinearRegression()
#lin_reg.fit(X, y)

# Fitting the Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(([[6.5]]))

# Visualizing the Random Forest Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth of Bluff (Random Forest Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
