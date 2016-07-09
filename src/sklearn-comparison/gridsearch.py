import numpy as np
import pandas as pd
import math
from datetime import datetime

from sklearn.grid_search import GridSearchCV

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from data_read import load

def grid_search(model,tuned_parameters,X,Y):
    startTime = datetime.now()
    model_gs = GridSearchCV(model, 
	tuned_parameters, cv=10)
    model_gs.fit(X, Y.values.ravel())
    print("Best parameters set found on development set:")
    print(model_gs.best_params_)
    print "\n"+"Task Completed! Completion time: "+ str(datetime.now()-startTime)
    return


# Load data
X, Y, P, Q = load()

rf_pg = [{'n_estimators':[1000,2000,3000,4000,5000],'random_state':[0]}]

svm_pg = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear'], 'random_state':[0]},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 
  'kernel': ['rbf'], 'random_state':[0]},
 ]

gbc_pg = [{'n_estimators':[1000,2000,3000,4000,5000],
'learning_rate':[100,10,1,1e-2,1e-3,1e-4,1e-5],
'max_depth':[10,100,1000],
'min_samples_split':[5,10,15,20],
'random_state':[0]}]

mlp_pg = [{'algorithm': ['adam','sgd','l-bfgs'], 
  'alpha': [1,1e-1,1e-2,1e-3, 1e-4,1e-5],
  'hidden_layer_sizes': [(100,), (200,), (300,), (400,),(500,)],
  'max_iter':[500], 'random_state':[0],
  'learning_rate' : ['constant', 'invscaling', 'adaptive']}]

# Grid search
# No grid search for Gaussian Naive Bayes
grid_search(RandomForestClassifier(),rf_pg,X,Y)
# {'n_estimators': 1000, 'random_state': 0}
# Time: 0:05:01.414756
print ("RF complete")
grid_search(SVC(),svm_pg,X,Y)
#{'kernel': 'rbf', 'C': 10, 'random_state': 0, 'gamma': 0.001}
# Time: 0:00:00.334069
print ("SVM complete")
grid_search(GradientBoostingClassifier(),gbc_pg,X,Y)
#{'min_samples_split': 5, 'n_estimators': 1000, 'learning_rate': 1, 'random_state': 0, 'max_depth': 10}
# Time: 1:08:32.784953
print ("GBC complete")
grid_search(MLPClassifier(),mlp_pg,X,Y)
#{'algorithm': 'adam', 'hidden_layer_sizes': (400,), 'learning_rate': 'constant', 
#'max_iter': 500, 'random_state': 0, 'alpha': 0.01}
# Time: 0:24:42.396666
print ("MLP complete")
