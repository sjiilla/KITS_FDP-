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

#cv accuracy for bagged tree ensemble
dt = tree.DecisionTreeClassifier()
bt2 = ensemble.BaggingClassifier(estimator = dt, n_estimators = 5)
bag_grid = {'criterion':['entropy','gini']}

bag_grid_estimator = model_selection.GridSearchCV(bt2, bag_grid, n_jobs=6)
bt2.fit(X_train, y_train)
scores = model_selection.cross_val_score(bt2, X_train, y_train, cv = 10)
print(scores)
print(scores.mean())

os.chdir("C:/Users/S Jilla/")
n_tree = 0
for est in bt2.estimators_: 
    dot_data = io.StringIO()
    #tmp = est.tree_
    tree.export_graphviz(est, out_file = dot_data, feature_names = X_train.columns)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())#[0] 
    graph.write_pdf("BagTree" + str(n_tree) + ".pdf")
    n_tree = n_tree + 1
    
os.getcwd()
