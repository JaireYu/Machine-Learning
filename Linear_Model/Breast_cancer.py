#coding=utf-8
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
dataset = load_breast_cancer()
print(len(dataset.data)) #共569数据使用500组作为训练数据剩余69作为测试
clf = linear_model.SGDClassifier(loss="log", max_iter=1000, tol=1e-3)
clf.fit(dataset.data[0:500], dataset.target[0:500])
proba_pred = clf.predict_proba(dataset.data[500:569])
label_pred = clf.predict(dataset.data[500:569])
pos_pred = []
for i in range(69):
    pos_pred.append(proba_pred[i][1])
fpr, tpr, thresholds = metrics.roc_curve(dataset.target[500:569], pos_pred, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
print(roc_auc)
print(metrics.classification_report(y_true=dataset.target[500:569], y_pred=label_pred))
plt.plot(fpr, tpr, lw=1, label='ROC(area = %0.2f)' % (roc_auc))
plt.xlabel("FPR (False Positive Rate)")
plt.ylabel("TPR (True Positive Rate)")
plt.title("Receiver Operating Characteristic, ROC(AUC = %0.2f)"% (roc_auc))
plt.show()

