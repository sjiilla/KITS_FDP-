import os
import pandas as pd
from sklearn import tree
from sklearn import model_selection
from sklearn import ensemble #This is what we introduced here.
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

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

#cv accuracy for bagged tree ensemble
dt = tree.DecisionTreeClassifier()
#Appy ensemble.BaggingClassificatier
#Base_Estimator = dt, n_estimators = 5(no. of trees)
#bt1 = ensemble.BaggingClassifier(estimator = dt, n_estimators = 3)
#bt1.fit(X_train, y_train)
#scores = model_selection.cross_val_score(bt1, X_train, y_train, cv = 10)
#print(scores)
#print(scores.mean())

#Alternative way with parameters and use GridSearchCV instead of cross_val_score
bt2 = ensemble.BaggingClassifier(estimator = dt, n_estimators = 3)
bag_grid = {'criterion':['entropy','gini']}

bag_grid_estimator = model_selection.GridSearchCV(bt2, bag_grid, n_jobs=6)
bt2.fit(X_train, y_train)
scores = model_selection.cross_val_score(bt2, X_train, y_train, cv = 10)
print(scores)
print(scores.mean())
