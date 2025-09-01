import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

#dataset
X = pd.read_csv('prosody_modified.csv')
X = X.drop('participant&question', axis=1 )

y = pd.read_csv('turker_scores.csv')
# y = y.drop('Participant', axis=1 )

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
y1 = y_train.iloc[0:,5]
yt1= y_test.iloc[0:,5]
 
# # Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1000, random_state = 0)
regressor.fit(X_train, y1)

# Predicting a new result
y_pred = regressor.predict(X_test)
print(regressor.score(X_train,y1))
print(regressor.score(X_test,yt1))
