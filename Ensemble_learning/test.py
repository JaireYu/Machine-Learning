import Adaboost
from numpy import *
DataMat = [[1.0, 2.1],[2.0, 1.1], [1.3, 1.0], [1.0, 1.0], [2.0, 1.0]]
ClassLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
D= mat(ones(shape = (5, 1), dtype = float32)/5)
Classifier = Adaboost.AdaBoostTrain(DataMat, ClassLabels)
print(Classifier)
