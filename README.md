# Salary-Prediction

#importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Creating Dataframes and seperating vriable

dataset= pd.read.csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#Train Set & test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =
train_test_split(X,y , test size = 1/3, random_state=0)

#fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X-train, y_train)

#Predictthetestresults

y_pred = regressor.predict(X_test)

#visualising the Training result
