"""Kaggle competition: Predicting a Biological Response.

Blending {RandomForests, ExtraTrees, GradientBoosting} + stretching to
[0,1]. The blending scheme is related to the idea Jose H. Solorzano
presented here:
http://www.kaggle.com/c/bioresponse/forums/t/1889/question-about-the-process-of-ensemble-learning/10950#post10950
'''You can try this: In one of the 5 folds, train the models, then use
the results of the models as 'variables' in logistic regression over
the validation data of that fold'''. Or at least this is the
implementation of my understanding of that idea :-)

The predictions are saved in test.csv. The code below created my best
submission to the competition:
- public score (25%): 0.43464
- private score (75%): 0.37751
- final rank on the private leaderboard: 17th over 711 teams :-)

Note: if you increase the number of estimators of the classifiers,
e.g. n_estimators=1000, you get a better score/rank on the private
test set.

Copyright 2012, Emanuele Olivetti.
BSD license, 3 clauses.

Modified by Ali A. Faruqi
Original script at:
https://github.com/emanuele/kaggle_pbr/blob/master/blend.py
"""

import numpy as np
import pandas as pd
import math

from data_read import load

from sklearn.cross_validation import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression



if __name__ == '__main__':
    np.random.seed(0) # seed to shuffle the train set
    n_folds = 10
    # X_train, y_train, X_test
    # X, Y, P, Q
    X, y, X_submission, y_test_p = load()
    # Create startifited k-fold data
    skf = list(StratifiedKFold(y.values.ravel(), n_folds,random_state=0))
    # Classifiers list with parameters
    clfs = [RandomForestClassifier(
    n_estimators=1000,random_state=0),
    SVC(kernel='rbf',C=10,
    gamma=0.001,random_state=0,probability=True),
    GradientBoostingClassifier(n_estimators=1000, learning_rate=1,
    max_depth=10, random_state=0, min_samples_split=5),
    GaussianNB(),
    MLPClassifier(algorithm='adam', alpha=0.01, max_iter=500,
    learning_rate='constant', hidden_layer_sizes=(400,), 
    random_state=0, learning_rate_init=1e-2,
    activation='logistic')]

    print "Creating train and test sets for blending."
    
    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))
    
    for j, clf in enumerate(clfs):
        #print j, clf
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            print "Fold", i
            # .iloc selects rows by index numbers
            X_train = X.iloc[train]
            y_train = y.iloc[train]
            X_test = X.iloc[test]
            y_test = y.iloc[test]
            clf.fit(X_train, y_train.values.ravel())
            y_submission = clf.predict_proba(X_test)[:,1]
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:,1]
        dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)

    print "Blending."
    clf = LogisticRegression()
    clf = clf.fit(dataset_blend_train, y.values.ravel())
    print ("Accuracy of Blended Classifiers: "+str(clf.score(dataset_blend_test,y_test_p)))