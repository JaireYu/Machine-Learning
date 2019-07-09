#coding=utf-8
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
def main():
    dataset = load_boston()
    while(1):
        vec = []
        for i in range(len(dataset.data[0])):
            x = input("input the data of {} {}".format(str(i), dataset.feature_names[i]))
            vec.append(x)
        print("伪逆矩阵模型:预测值和参数\n")
        print(predict(dataset, vec))
        print("sklearnlinearRegression模型:预测值和参数\n")
        print(skpredict(dataset, vec))

def predict(dataset, vec):
    ONE = np.ones([len(dataset.data),1], dtype= float)
    X = np.concatenate((dataset.data, ONE), axis=1)
    y = np.reshape(dataset.target,(len(dataset.target), 1))
    w = (np.mat(X).T * np.mat(X)).I * np.mat(X.T) * np.mat(y)
    num = 0.0
    w = list(w)
    print(w)
    newvec = vec[:] #注意列表是引用的, 不能直接append否则会修改全空间的vec
    newvec.append(1.0)
    for i in range(len(w)):
        num += newvec[i]*float(w[i])
    return num, w

def skpredict(dataset, vec):
    X = dataset.data
    Y = dataset.target
    LR = LinearRegression()
    LR.fit(X, Y)
    vec = np.reshape(vec, (1, len(vec)))
    return LR.predict(vec), LR.coef_, LR.intercept_

