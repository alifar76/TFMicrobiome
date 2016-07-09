from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import numpy as np
import pandas as pd
import math
from datetime import datetime

from data_read import load

# Load data
X, Y, P, Q = load()


clf = RandomForestClassifier(
	n_estimators=1000,random_state=0).fit(X, Y.values.ravel())
print ("Accuracy of Random Forest Classifier: "+str(clf.score(P,Q)))

clf2 = SVC(kernel='rbf',C=10,
	gamma=0.001,random_state=0).fit(X, Y.values.ravel())
print ("Accuracy of SVM: "+str(clf2.score(P,Q)))


clf3 = GradientBoostingClassifier(n_estimators=1000, learning_rate=1,
max_depth=10, random_state=0, min_samples_split=5).fit(X, Y.values.ravel())
print ("Accuracy of Gradient Boosting Classifier: "+str(clf3.score(P,Q)))

clf4 = GaussianNB().fit(X, Y.values.ravel())
print ("Accuracy of Gaussian Naive Bayes Classifier: "+str(clf4.score(P,Q)))


# algorithm, learning_rate_init, alpha, hidden_layer_sizes 
# and activation have impact
clf6 = MLPClassifier(algorithm='adam', alpha=0.01, max_iter=500,
	learning_rate='constant', hidden_layer_sizes=(400,), 
	random_state=0, learning_rate_init=1e-2,
	activation='logistic').fit(X, Y.values.ravel())
print ("Accuracy of Multi-layer Perceptron Classifier: "+str(clf6.score(P,Q)))