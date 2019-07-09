#coding=utf-8
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

colors = ['navy', 'turquoise', 'darkorange']

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):      #将三元数组捆绑成zip遍历三次即可
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                label=target_name)  #这里是选择y==i的行的第0列相当于一种切片方式
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of IRIS dataset')

plt.show()
print(X_r2)
print(X_r2[y==0][0])
print(X_r2[y==0, 0])
print(X_r2[[1,2,3]])