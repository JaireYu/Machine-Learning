#coding=utf-8
from numpy import *
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def pca(dataMat, topN):
    MeanVals = mean(dataMat, axis=0) #对列求均值
    DeMeanMat = dataMat - MeanVals  #去均值化
    CovMat = cov(DeMeanMat, rowvar= False) #以列为数据计算协方差矩阵
    EigVals, EigVects = linalg.eig(mat(CovMat)) #求特征值特征向量
    EigValsIndex = argsort(EigVals) #返回由小到大的索引值
    EigValsIndex = EigValsIndex[:-(topN + 1):-1] #取最大的TopN个
    UsedEigVects = EigVects[:,EigValsIndex] #选择对应的列(特征向量
    LowDDataMat = DeMeanMat * UsedEigVects #XE
    ReorganDataMat = LowDDataMat * UsedEigVects.T + MeanVals #如果正确的话就是DataMat
    return LowDDataMat, ReorganDataMat



