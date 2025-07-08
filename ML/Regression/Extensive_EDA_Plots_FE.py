# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 09:35:56 2025

@author: S Jilla
"""

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')

#import xgboost as xgb  #XGBM algorithm
#from xgboost import XGBRegressor

os.chdir("C:/Data Science/House/")

#Load the data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
train_data.info()
print(train_data.shape)
print(test_data.shape)

#Number columns are picked and Id, SalePrice are removed from the list.
previous_num_columns = train_data.select_dtypes(exclude=['object']).columns.values.tolist()
previous_num_columns.remove('Id')
previous_num_columns.remove('SalePrice')
print(previous_num_columns)

#Delete Outlier Data
#Check LotFrontage, LotArea, MasVnrArea.... column trend in Train and Test,.
#KDE plot: Kernel Density Estimate plot
sns.kdeplot(train_data['LotFrontage'])
sns.kdeplot(test_data['LotArea'], color="r")

#sharey: Share Y Axis
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey=False)
sns.kdeplot(train_data['LotFrontage'], ax=ax1)
sns.kdeplot(train_data['MasVnrArea'], ax=ax2, color="g")
sns.kdeplot(test_data['LotArea'], ax=ax3,color="r")

#Now check the data -- how many records are there with > 1500 in train and test -- Compare the same with the above plot
print('train:', train_data['LotFrontage'][train_data['LotFrontage'] > 200].shape)
print('train:', train_data['LotArea'][train_data['LotArea'] > 60000].shape)
print('train:', train_data['MasVnrArea'][train_data['MasVnrArea'] > 1500].shape)

#==============================================================================
# print('test:', test_data['LotFrontage'][test_data['LotFrontage'] > 200].shape)
# print('test:', test_data['LotArea'][test_data['LotArea'] > 60000].shape)
# print('test:', test_data['MasVnrArea'][test_data['MasVnrArea'] > 1500].shape)
# 
#==============================================================================
print(train_data.shape)
train_data.drop(train_data[train_data["LotFrontage"] > 200].index, inplace=True)
train_data.drop(train_data[train_data["LotArea"] > 60000].index, inplace=True)
train_data.drop(train_data[train_data["MasVnrArea"] > 1500].index, inplace=True)
print(train_data.shape)

#The record count is stored -- To separate Train and Test from Combined Data
train_length = train_data.shape[0]

#Both Train and Test are combined to perform EDA and FE
combined_data = pd.concat([train_data.loc[:, : 'SalePrice'], test_data])
combined_data = combined_data[test_data.columns]
print(combined_data.shape)

#Filling missing Values
#missing data columns are obtained -- 
has_null_columns = combined_data.columns[combined_data.isnull().any()].tolist()
print(has_null_columns)
#A function to fill the missing data with given value
def fill_missing_combined_data(column, value):
    combined_data.loc[combined_data[column].isnull(),column] = value
    
#Lot Frontage -- Filled with median of the Neighborhood, grouped
combined_data['LotFrontage'].groupby(combined_data["Neighborhood"]).median().plot()
combined_data['LotFrontage'].groupby(combined_data["Neighborhood"]).mean().plot()

#You can also group by multiple columns
lf_neighbor_map = combined_data['LotFrontage'].groupby([combined_data["Neighborhood"], combined_data["HouseStyle"]]).median()
print(lf_neighbor_map)

#Get the records with missing values and fill median of Neighborhood group.
rows = combined_data['LotFrontage'].isnull()
combined_data['LotFrontage'][rows] = combined_data['Neighborhood'][rows].map(lambda neighbor : lf_neighbor_map[neighbor])

#Check the LotFrontage missing data records -- which is now zero
combined_data[combined_data['LotFrontage'].isnull()]

#Alley -- All the missing values are filled with NA, which means No Alley
combined_data.shape
train_data[train_data['Alley'].isnull()].shape
combined_data[combined_data['Alley'].isnull()].shape
fill_missing_combined_data('Alley', 'NA')

#FireplaceQu - For Fireplaces 0, FireplaceQu is set to NA, indicating No Fireplace, which is the case of missing 1420 records of data
combined_data[combined_data['FireplaceQu'].isnull()].shape
fill_missing_combined_data('FireplaceQu', 'NA')

combined_data[combined_data['PoolQC'].isnull()].shape
fill_missing_combined_data('PoolQC', 'NA')

combined_data[combined_data['MiscFeature'].isnull()].shape
fill_missing_combined_data('MiscFeature', 'NA')

combined_data[combined_data['Fence'].isnull()].shape
fill_missing_combined_data('Fence', 'NA')

#Fill isnull(N/A) records with 'None' in this case
combined_data[combined_data['MasVnrArea'].isnull()].shape
combined_data['MasVnrType'].fillna('None', inplace=True)
#Fill isnull(N/A) records with 0 in this case
combined_data['MasVnrArea'].fillna(0, inplace=True)

#Basement columns -- BsmtQual / BsmtCond / BsmtExposure / BsmtFinType1 / BsmtFinType2
basement_cols=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2']

#The data for the missing string type is filled with NA, which means No Basement
for column in basement_cols:
    if 'FinSF' not in column:
        #print(column)
        fill_missing_combined_data(column, 'NA')
    else:
        print(column)
        fill_missing_combined_data(column, 0)
        
#Few other Basement related continous columns are filled with 0 -- which means no basement
fill_missing_combined_data('BsmtUnfSF', 0)
fill_missing_combined_data('TotalBsmtSF', 0)
fill_missing_combined_data('BsmtFullBath', 0)
fill_missing_combined_data('BsmtHalfBath', 0)

#Garage Columns
garage_cols=['GarageType','GarageQual','GarageCond','GarageYrBlt','GarageFinish','GarageCars','GarageArea']

#The data for the missing string type is filled with NA, which means No Garage
for column in garage_cols:
    if column != 'GarageCars' and column != 'GarageArea':
        fill_missing_combined_data(column, 'NA')
    else:
        fill_missing_combined_data(column, 0)

#Check if there is any missing data
has_null_columns = combined_data.columns[combined_data.isnull().any()].tolist()
print(has_null_columns)
       
#Fill with Mode
#Electrical - Missing a piece of data, filled with the highest number of occurrences.
#sns.countplot(combined_data['Utilities'], data=combined_data) #To get the most frequent value which is Mode
fill_missing_combined_data('MSZoning', 'RL')
fill_missing_combined_data('Utilities', 'AllPub')
fill_missing_combined_data('Exterior1st', 'VinylSd')
fill_missing_combined_data('Exterior2nd', 'VinylSd')
fill_missing_combined_data('Electrical', 'SBrkr')
fill_missing_combined_data('KitchenQual', 'TA')
fill_missing_combined_data('Functional', 'Typ')
fill_missing_combined_data('SaleType', 'WD')

#Check if there is any missing data -- It should be empty now
has_null_columns = combined_data.columns[combined_data.isnull().any()].tolist()
print(has_null_columns)
##End of EDA
