# Importing the libraries
import numpy as np # for array operations
import pandas as pd # for working with DataFrames

# scikit-learn modules
from sklearn.model_selection import train_test_split # for splitting the data
from sklearn.metrics import mean_squared_error # for calculating the cost function
from sklearn.ensemble import RandomForestRegressor # for building the model

from sklearn.metrics import mean_squared_error # for calculating the cost function
import pickle

# For creating a pmml file
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline

from scikit2pmml import scikit2pmml

# Reading the data
dataset = pd.read_csv('boston_train.csv')

x = dataset.drop('MEDV', axis = 1) # Features
y = dataset['MEDV']  # Target

# Splitting the dataset into training and testing set (80/20)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)

# Initializing the Random Forest Regression model with 10 decision trees
model = RandomForestRegressor(n_estimators = 100, random_state = 20)

# Fitting the Random Forest Regression model to the data
model.fit(x_train, y_train) 

# Predicting the target values of the test set
y_pred = model.predict(x_test)
#print(y_pred)

# RMSE (Root Mean Square Error)
rmse = float(format(np.sqrt(mean_squared_error(y_test, y_pred)),'.3f'))
#print("\nRMSE:\n",rmse)

# Save the model in pickle format
pickle.dump(model, open('model.pkl', 'wb'))







