#Create correlation matix using seabon-->Heatmap
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
os.chdir("C:/Data Science/House/")

#Load the data
train_data = pd.read_csv("train.csv")

corrMatrix=train_data[["SalePrice","OverallQual","GrLivArea","GarageCars",
                  "GarageArea","GarageYrBlt","TotalBsmtSF","1stFlrSF","FullBath",
                  "TotRmsAbvGrd","YearBuilt","YearRemodAdd"]].corr()

sns.set(font_scale=1.10)
plt.figure(figsize=(10, 10))

sns.heatmap(corrMatrix, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='viridis',linecolor="white")
plt.title('Correlation between features');
