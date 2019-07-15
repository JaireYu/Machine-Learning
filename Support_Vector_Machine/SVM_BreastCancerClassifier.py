from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
import numpy as np
BCD = load_breast_cancer()
X = BCD.data
X1 = X[0: 2*len(X)/3]
Y = BCD.target
Y1 = Y[0: 2*len(Y)/3]
clf = SVC(C=1.0, kernel='poly', degree= 2, gamma='auto')
clf.fit(X1, Y1)
Res = clf.predict(X[len(X1): len(X)])
count = 0
for i in range(len(Y)-len(Y1)):
    if(Res[i] != Y[len(Y1) + i]):
        count += 1
print("The error rate is {}".format(str(float(count)/(len(X) - len(X1)))))