import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression

#dataset
X = pd.read_csv('p_train.csv')
X = X.drop('participant&question', axis=1 )

y = pd.read_csv('t_train.csv')

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.0)

#You can change 19 to whatever column number you want
#You can automate if you want for many columns
 
y1 = y_train.iloc[0:,19]

mi = mutual_info_regression(X_train,y1)

for i in mi:
	print("{},".format(i),end='')
