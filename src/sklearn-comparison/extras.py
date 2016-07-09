from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso

from sklearn.ensemble import AdaBoostClassifier

from data_read import load

# Load data
X, Y, P, Q = load()

## Ensemble Voting

cl1 = RandomForestClassifier(
	n_estimators=1000,random_state=0)
cl2 = SVC(kernel='rbf',C=10,
	gamma=0.001,random_state=0,probability=True)
cl3 = GradientBoostingClassifier(n_estimators=1000, learning_rate=1,
max_depth=10, random_state=0, min_samples_split=5)
cl4 = GaussianNB()
cl5 = MLPClassifier(algorithm='adam', alpha=0.01, max_iter=500,
	learning_rate='constant', hidden_layer_sizes=(400,), 
	random_state=0, learning_rate_init=1e-2,
	activation='logistic')


eclf1 = VotingClassifier(estimators=[
('rf', cl1), ('svc', cl2), ('gbc', cl3),
('gnb',cl4),('mlp',cl5)
], voting='hard')

eclf1 = eclf1.fit(X, Y.values.ravel())
print ("Accuracy of Voting Ensemble: "+str(eclf1.score(P,Q)))



clf5 = SGDClassifier(loss="perceptron", penalty="elasticnet", 
	random_state=0).fit(X, Y.values.ravel())
print ("Accuracy of SGDClassifier: "+str(clf5.score(P,Q)))

gbc = GradientBoostingClassifier(loss='exponential').fit(X, Y.values.ravel())
adaboost = AdaBoostClassifier(n_estimators=10000, learning_rate=100).fit(X, Y.values.ravel())
print ("Accuracy of GBC: "+str(gbc.score(P,Q)))
print ("Accuracy of Adaboost: "+str(adaboost.score(P,Q)))


### Calculate MSE of different models
rf = clf.predict(P)
svmp = clf2.predict(P)
gbc = clf3.predict(P)
gnbc = gnb.predict(P)
sgdc = clf5.predict(P)
enc = preprocessing.LabelEncoder()
J = enc.fit_transform(Q)
print ("MSE of RF: "+str(mean_squared_error(J,enc.fit_transform(rf))))
print ("MSE of SVM: "+str(mean_squared_error(J,enc.fit_transform(svmp))))
print ("MSE of Gradient Boost: "+str(mean_squared_error(J,enc.fit_transform(gbc))))
print ("MSE of Naive Bayes: "+str(mean_squared_error(J,enc.fit_transform(gnbc))))
print ("MSE of SGD: "+str(mean_squared_error(J,enc.fit_transform(sgdc))))