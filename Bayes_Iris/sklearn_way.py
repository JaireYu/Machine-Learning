from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
import numpy as np
clf = GaussianNB()
iris = datasets.load_iris()
dataset = iris.data[0:49]
dataset = np.concatenate((dataset, iris.data[50:99]), axis=0)
dataset = np.concatenate((dataset, iris.data[100:149]), axis=0)
target = iris.target[0:49]
target = np.concatenate((target, iris.target[50:99]),axis=0)
target = np.concatenate((target, iris.target[100:149]), axis=0)
print(dataset)
print(target)
clf.fit(dataset, target)
print(clf.theta_)
print(clf.sigma_)
print(clf.predict_proba([[5.9,3.0,5.1,1.8]]))
print(clf.predict_proba([[5.7,2.8,4.1,1.3]]))
print(clf.predict_proba([[5.0,3.3,1.4,0.2]]))

