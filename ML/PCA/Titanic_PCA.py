# -*- coding: utf-8 -*-
"""
@author: Sreenivas.J
"""
import os
import pandas as pd
from sklearn import decomposition, preprocessing, tree
#returns current working directory
os.getcwd()
#changes working directory
os.chdir("C:/Users/SJilla/")

titanic_train = pd.read_csv("train.csv")
titanic_test = pd.read_csv("test.csv")

#EDA
titanic_train.shape
titanic_train.info()

titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'Sex', 'Embarked'])
titanic_train1.shape
titanic_train1.info()

X_train = titanic_train1.drop(['PassengerId','Age','Cabin','Ticket', 'Name','Survived'],  axis=1, inplace=False)
X_train.info()
X_train.shape

#Here comes the PCA!
scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
scaled_data = scaler.transform(X_train)
print(X_train)
print(scaled_data)

pca = decomposition.PCA(n_components=4)
pca.fit(scaled_data)

print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())

#Transformation of PCA happens here
transformed_X_train = pca.transform(scaled_data)

transformed_X_train.shape
print(transformed_X_train)
y_train = titanic_train['Survived']

dt = tree.DecisionTreeClassifier()
#dt.fit(X_train, y_train) #On original data columns
dt.fit(transformed_X_train, y_train) #On Transoformed columns

titanic_test.info() #Found that one row has Fare = null in test data. Instead of dropping this column, let's take the mean of it.
titanic_test.Fare[titanic_test['Fare'].isnull()] = titanic_test['Fare'].mean()

#Now apply same get_dummies and drop columns on test data as well like above we did for train data
titanic_test1 = pd.get_dummies(titanic_test,columns=['Pclass', 'Sex', 'Embarked'])
X_titanic_test = titanic_test1.drop(['PassengerId','Age','Cabin','Ticket', 'Name'],  axis=1, inplace=False)

scaler = preprocessing.StandardScaler()
scaler.fit(X_titanic_test)
scaled_test_data = scaler.transform(X_titanic_test)

pca = decomposition.PCA(n_components=4)
pca.fit(X_titanic_test)

X_transformed_Test = pca.transform(scaled_test_data)

#Apply the model on Furture/Test data
os.getcwd()
titanic_test['Survived'] = dt.predict(X_transformed_Test)
os.chdir("C:/Users/S Jilla/")
titanic_test.to_csv("Submission_PCA.csv",columns=['PassengerId','Survived'],index=False)
