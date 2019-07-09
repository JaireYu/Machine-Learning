#coding=utf-8
import PCA_myway
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np

iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

Dimen2Data, OriginalData = PCA_myway.pca(X, 2)

fig = plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2
# 我的模型
Dimen2Data = Dimen2Data.getA()
ColorForMe = []
for i in y:
    ColorForMe.append(colors[i])
print(Dimen2Data)
ax1 = fig.add_subplot(211)

ax1.scatter(Dimen2Data[:,0], Dimen2Data[:,1], color = ColorForMe, alpha=.8, lw=lw)
ax1.legend(loc='best', shadow=False, scatterpoints=1)
ax1.set_title('My PCA of IRIS dataset')
#sklearn pca

pca_sklearn = PCA(n_components=2)
X_r = pca_sklearn.fit(X).transform(X)

ax2 = fig.add_subplot(212)
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    ax2.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
ax2.legend(loc='best', shadow=False, scatterpoints=1)
ax2.set_title('sklearn PCA of IRIS dataset')

plt.show()