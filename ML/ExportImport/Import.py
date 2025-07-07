# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 15:19:13 2025

@author: S Jilla
"""

import os
import pandas as pd
#import sklearn.externals
import joblib

#changes working directory
os.chdir("C:/Users/SJilla/")

#predict the outcome using decision tree
titanic_test = pd.read_csv("test.csv")
titanic_test.Fare[titanic_test['Fare'].isnull()] = titanic_test['Fare'].mean()

titanic_test1 = pd.get_dummies(titanic_test, columns=['Pclass', 'Sex', 'Embarked'])
X_test = titanic_test1.drop(['PassengerId','Age','Cabin','Ticket', 'Name'], axis=1, inplace=False)

#Use load method to load Pickle file
os.chdir("C:/Users/S Jilla/")
dtree = joblib.load("TitanicVer1.pkl")
titanic_test['Survived'] = dtree.predict(X_test)
titanic_test.to_csv("submissionUsingJobLib.csv", columns=['PassengerId','Survived'], index=False)
