#coding=utf-8
import numpy as np
from math import *
import Iris
from sklearn import datasets
iris = datasets.load_iris()
dataset = iris.data
labels = iris.target
targetnames = iris.target_names
featurenames = iris.feature_names
ParameterMat, PC = Iris.CalEverageAndDeviation(dataset, labels)
print(ParameterMat)
while(1):
    X = []
    for i in range(4):
        x = input("What's the number of {}?".format(featurenames[i]))
        X.append(x)
    Prob = Iris.Probability(ParameterMat, X, PC)
    print (Prob)
    print(Iris.SelectLabel(targetnames, Prob))
