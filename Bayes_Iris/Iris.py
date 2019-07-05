#coding=utf-8
import numpy as np
from math import *
from sklearn import datasets

def CalEverageAndDeviation(dataset, target):    #计算某标签下的属性数据集的方差均值矩阵i为属性j为标签
    dataset = np.ravel(dataset)
    Res = np.empty([3, 4 ,2], dtype=float)
    PC = [0,0,0]
    target = list(target)
    size = len(dataset)
    dataset1 = dataset[0: size: 4]  # 第一个属性的数据
    dataset2 = dataset[1: size: 4]
    dataset3 = dataset[2: size: 4]
    dataset4 = dataset[3: size: 4]
    dataset = [dataset1, dataset2, dataset3, dataset4]
    for i in range(3):
        for j in range(4):
            if i == 2:
                begin = target.index(2)
                end = len(target)
            else:
                begin = target.index(i)
                end = target.index(i + 1)
            Res[i][j][0] = np.mean(dataset[j][begin: end-1])    #留下一个作为验证
            Res[i][j][1] = np.std(dataset[j][begin: end-1], ddof=1)
            PC[i] = (end - begin)/float(len(target))
    return Res, PC

def Probability(Res, x, PC): #根据参数返回xi对应的log概率res是参数矩阵, PC = p(C) ，x是数据向量，返回概率矩阵
    Prob = []
    for i in range(3):
        prob = 0
        for j in range(4):
            prob += -(x[j] - Res[i][j][0])**2/2/float(Res[i][j][1]**2) - 0.5*log(2*pi) - log(Res[i][j][1])
        prob += log(PC[i])
        Prob.append(exp(prob))
    return Prob

def SelectLabel(names, Prob):
    num = max(Prob)
    return names[Prob.index(num)]

