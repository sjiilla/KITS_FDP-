import pandas as pd
import os
from sklearn import impute
from sklearn import ensemble, tree
from sklearn import model_selection

#changes working directory
os.chdir("C:/Users/SJilla/")

titanic_train = pd.read_csv("train.csv")
titanic_train.shape
titanic_train.info()

titanic_test = pd.read_csv('test.csv')
titanic_test.shape
titanic_test.info()
titanic_test.Survived = None

#it gives the same never of levels for all the categorical variables
titanic = pd.concat([titanic_train, titanic_test])

#create title column from name
def extract_title(name):
     return name.split(',')[1].split('.')[0].strip()
titanic['Title'] = titanic['Name'].map(extract_title)

#create an instance of Imputer class with required arguments
mean_imputer = impute.SimpleImputer()
#compute mean of age and fare respectively
mean_imputer.fit(titanic_train[['Age','Fare']])
#fill up the missing data with the computed means 
titanic[['Age','Fare']] = mean_imputer.transform(titanic[['Age','Fare']])

#creaate categorical age column from age
def convert_age(age):
    if(age >= 0 and age <= 10): 
        return 'Child'
    elif(age <= 25): 
        return 'Young'
    elif(age <= 50): 
        return 'Middle'
    else: 
        return 'Old'
titanic['Age1'] = titanic['Age'].map(convert_age)

titanic['FamilySize'] = titanic['SibSp'] +  titanic['Parch'] + 1
def convert_familysize(size):
    if(size == 1): 
        return 'Single'
    elif(size <=3): 
        return 'Small'
    elif(size <= 6): 
        return 'Medium'
    else: 
        return 'Large'
titanic['FamilySize1'] = titanic['FamilySize'].map(convert_familysize)

#convert categorical columns to one-hot encoded columns
titanic1 = pd.get_dummies(titanic, columns=['Sex','Pclass','Embarked', 'Age1', 'Title', 'FamilySize1'])
titanic1.shape
titanic1.info()

titanic2 = titanic1.drop(['PassengerId','Name','Age','Ticket','Cabin','Survived'], axis=1, inplace=False)
titanic2.shape

X_train = titanic2[0:titanic_train.shape[0]]
X_train.shape
X_train.info()
y_train = titanic_train['Survived']

#create estimators for voting classifier
dt_estimator = tree.DecisionTreeClassifier(random_state=100)
rf_estimator = ensemble.RandomForestClassifier(random_state=100)
ada_estimator = ensemble.AdaBoostClassifier(random_state=100)

#Soft voting with Weights
voting_model = ensemble.VotingClassifier(estimators=[('dt', dt_estimator), ('rf', rf_estimator), ('ada', ada_estimator)], voting='soft', weights=[2,1,2])

#Parameters to above 3 models
voting_params = {'dt__max_depth':[3,5,7], 'rf__n_estimators':[50], 'rf__max_features':[5,6], 'rf__max_depth':[5], 'ada__n_estimators':[50]}
grid_voting_estimator = model_selection.GridSearchCV(voting_model, voting_params, cv=10, n_jobs=6)
grid_voting_estimator.fit(X_train, y_train)
print(grid_voting_estimator.cv_results_)
print(grid_voting_estimator.best_score_)
print(grid_voting_estimator.best_params_)
print(grid_voting_estimator.score(X_train, y_train))

X_test = titanic2[titanic_train.shape[0]:]
X_test.shape
X_test.info()

titanic_test['Survived'] = grid_voting_estimator.predict(X_test)
os.chdir("C:/Users/S Jilla/")
#Predictions using Voting classifier
titanic_test.to_csv('submissionVotingSoft.csv', columns=['PassengerId','Survived'],index=False)
