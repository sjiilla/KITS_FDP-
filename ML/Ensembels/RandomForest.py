# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 15:17:02 2025

@author: S Jilla
"""
import os, io, pydotplus
import pandas as pd
from sklearn import tree
from sklearn import model_selection
from sklearn import ensemble #This is what we introduced here.

#returns current working directory 
os.getcwd()
#changes working directory
os.chdir("C:/Users/SJilla/")
titanic_train = pd.read_csv("train.csv")

#EDA
titanic_train.shape
titanic_train.info()

titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'Sex', 'Embarked'])
titanic_train1.shape
titanic_train1.info()

X_train = titanic_train1.drop(['PassengerId','Age','Cabin','Ticket', 'Name','Survived'], axis=1, inplace=False)
y_train = titanic_train['Survived']

#Random Forest classifier
#Remember RandomForest works for Decission Trees only and there is NO Base_Estimator parameter exists
rf_estimator = ensemble.RandomForestClassifier(random_state=1)
#n_estimators: no.of trees to be built
#max_features: Maximum no. of features to try with
#rf_grid = {'n_estimators':list(range(200,251,50)),'max_features':[3,6,9],'criterion':['entropy','gini']}
rf_grid = {'n_estimators':[10],'max_features':[6],'criterion':['entropy','gini']}
rf = model_selection.GridSearchCV(rf_estimator,rf_grid, cv=10, n_jobs=10)
rf.fit(X_train, y_train)
#rf_grid_estimator.grid_scores_
rf.best_estimator_
rf.best_score_
#rf_grid_estimator.best_estimator_.feature_importances_
#rf.score(X_train, y_train)

titanic_test = pd.read_csv("test.csv")
titanic_test.shape
titanic_test.Fare[titanic_test['Fare'].isnull()] = titanic_test['Fare'].mean()

titanic_test1 = pd.get_dummies(titanic_test, columns=['Pclass', 'Sex', 'Embarked'])
titanic_test1.shape
titanic_test1.info()

os.chdir("C:/Users/S Jilla/")
X_test = titanic_test1.drop(['PassengerId','Age','Cabin','Ticket', 'Name'], axis=1, inplace=False)
titanic_test['Survived'] = rf.predict(X_test)
titanic_test.to_csv("submission_rf.csv", columns=['PassengerId','Survived'], index=False)
os.getcwd()

