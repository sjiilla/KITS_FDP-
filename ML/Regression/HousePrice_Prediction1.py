# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 11:26:13 2025

@author: S Jilla
"""
#Regression - House sale price prediction
import os
import pandas as pd
from sklearn import tree
from sklearn import metrics
from sklearn import model_selection
import numpy as np
#import io, pydotplus
import math
import seaborn as sns
import matplotlib.pyplot as plt

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("C:/Data Science/House/")

house_train = pd.read_csv("train.csv")
house_train.shape
house_train.info()

house_test = pd.read_csv("test.csv")
house_test.shape
house_test.info()

#Concatenate train and test data and we will seperate before applying the model
house_data = pd.concat([house_train, house_test],ignore_index=True)
house_data.drop(["Id","SalePrice"], axis=1, inplace=True)
house_data.shape
house_data.info()

#convert numerical columns to categorical type              
house_data['MSSubClass'] = house_data['MSSubClass'].astype('category')

#convert categorical columns to numeric type
ordinal_features1 = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "GarageQual", "GarageCond", "PoolQC", "FireplaceQu", "KitchenQual", "HeatingQC"]
#ordinal_features1 = [col for col in house_train if 'TA' in list(house_train[col])]
quality_dict = {None: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
for feature in ordinal_features1:
    null_idx = house_data[feature].isnull()
    house_data.loc[null_idx, feature] = None 
    house_data[feature] = house_data[feature].map(quality_dict)

#==============================================================================
# house_data.head(100)
# house_data['BsmtCond']
#==============================================================================

#handle missing data columns
#See for how many rows the data is missing!
total_missing = house_data.isnull().sum()

#for just dry run, go ahead and delete where the data is missing.
#This is just for a dry run to understand the next steps.
#in reality, we have impute these values
to_delete = total_missing[total_missing>0]
house_data.drop(list(to_delete.index), axis=1, inplace=True)
house_data.shape

#Numerical columns.. include=['number']
numeric_cols = house_data.select_dtypes(include=['number']).columns
#Categorical columns .. exclude = ['number'])
cat_cols = house_data.select_dtypes(exclude = ['number']).columns

#Automated to get all category columns and use one hot encoding instead of writing each and every column
house_data1 = pd.get_dummies(house_data, columns=cat_cols)
house_data1.shape

#Plot my sale price
sns.distplot(house_train['SalePrice'])
plt.show()

#splitting train data as conctenated in the begining
house_train1 = house_data1[:house_train.shape[0]]
#splitting test data as conctenated in the begining
house_test1 = house_data1[house_train.shape[0]:]
