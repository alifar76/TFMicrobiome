from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import numpy as np
import pandas as pd
import math

## Input file specific variables
otuinfile = 'lozupone_hiv.txt'
metadata = 'mapfile_lozupone.txt'
# Split 55% of data as training and 45% as test
train_ratio = 0.55
metavar = ['hiv_stat','HIV status']
levels = ['HIV_postive','HIV_negative','Undetermined']



a = pd.read_table(otuinfile,skiprows=1,index_col=0)
b = a.transpose()
response = {}
hiv = 0
infile = open(metadata,'rU')
for line in infile:
  if line.startswith("#SampleID"):
    spline = line.strip().split("\t")
    hiv = spline.index(metavar[0])
  else:
    spline = line.strip().split("\t")
    response[spline[0]] = spline[hiv]
u = [response[x] for x in list(b.index)]
v = [levels[0] if x == 'True' else levels[1] if x == 'False' else levels[2] for x in u]
b.loc[:,metavar[1]] = pd.Series(v, index=b.index)
c = b[b[metavar[1]].isin([levels[0], levels[1]])]
n_train = int(math.ceil(train_ratio*c.shape[0]))
train_dataset = pd.DataFrame()
test_dataset = pd.DataFrame()
train_dataset = c[:n_train]
test_dataset = c[n_train:]


# Train dataset
X = train_dataset.drop(metavar[1],1)
Y = train_dataset[[metavar[1]]]
# Test dataset
P = test_dataset.drop(metavar[1],1)
Q = test_dataset[[metavar[1]]]

clf = RandomForestClassifier(n_estimators=6000)
clf = clf.fit(X, Y.values.ravel())
print ("Accuracy of Random Forest Classifier: "+str(clf.score(P,Q)))

clf2 = svm.SVC(C=1000)
clf2 = clf2.fit(X, Y.values.ravel())
print ("Accuracy of SVM: "+str(clf2.score(P,Q)))

clf3 = GradientBoostingClassifier(n_estimators=10000, learning_rate=0.01,
max_depth=10, random_state=0).fit(X, Y.values.ravel())
print ("Accuracy of Gradient Boosting Classifier: "+str(clf3.score(P,Q)))

gnb = GaussianNB()
gnb = gnb.fit(X, Y.values.ravel())
print ("Accuracy of Gaussian Naive Bayes Classifier: "+str(gnb.score(P,Q)))

clf5 = SGDClassifier(loss="perceptron", penalty="elasticnet", 
random_state=0).fit(X, Y.values.ravel())
print ("Accuracy of SGDClassifier: "+str(clf5.score(P,Q)))
