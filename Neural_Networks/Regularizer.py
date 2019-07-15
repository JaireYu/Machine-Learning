import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math
BATCH_SIZE = 30
seed = 2
rdm = np.random.RandomState(seed)
X = rdm.randn(300,2)#三百组标准正态分布
Y_ = [int(x0*x0 + x1*x1<2) for (x0,x1) in X] #按照规则生成非线性数据集
Y_color = ["red" if y else "blue" for y in Y_]
fig = plt.figure()
X = np.vstack(X).reshape(-1,2)  #vstack其实没有用吧...
Y_ = np.vstack(Y_).reshape(-1,1)
ax1 = fig.add_subplot(221)
ax1.scatter(x=X[:,0], y=X[:,1], c=Y_color)
def get_weight(Shape, regularize):
    w = tf.Variable(tf.random_normal(Shape), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularize)(w))
    return w

def get_bias(Shape):
    b = tf.Variable(tf.constant(0.01, shape=Shape))
    return b

x = tf.placeholder(tf.float32, shape = (None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

w1 = get_weight([2, 11], 0.01)
b1 = get_bias([11])
y1 = tf.nn.relu(tf.matmul(x, w1) + b1)#修正线性单元

w2 = get_weight([11, 1], 0.01)
b2 = get_bias([1])
y = tf.matmul(y1, w2) + b2

loss_mse = tf.reduce_mean(tf.square(y-y_))
loss_total = loss_mse + tf.add_n(tf.get_collection("losses"))

train_step2 = tf.train.AdamOptimizer(0.0001).minimize(loss_total)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 80000
    for i in range(STEPS):
        start = (i*BATCH_SIZE)%300
        end = start + BATCH_SIZE
        sess.run(train_step2, feed_dict={x:X[start:end], y_:Y_[start:end]})
        if(i % 2000 == 0):
            loss_mse_v = sess.run(loss_mse, feed_dict={x:X, y_:Y_})
            print("STEP:{}, LOSS:{}".format(i, loss_mse_v))
    xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
    grid = np.c_[xx.ravel(), yy.ravel()] #生成网格数据点配对成为矩阵
    probs = sess.run(y, feed_dict={x: grid})
    probs = probs.reshape(xx.shape)
ax2 = fig.add_subplot(222)
ax2.scatter(X[:,0], X[:,1], c = Y_color)
ax2.contour(xx, yy, probs, levels = [.5])
ax2.set_title("With Regularizer")

train_step1 = tf.train.AdamOptimizer(0.0001).minimize(loss_mse)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 80000
    for i in range(STEPS):
        start = (i*BATCH_SIZE)%300
        end = start + BATCH_SIZE
        sess.run(train_step1, feed_dict={x:X[start:end], y_:Y_[start:end]})
        if(i % 2000 == 0):
            loss_mse_v = sess.run(loss_mse, feed_dict={x:X, y_:Y_})
            print("STEP:{}, LOSS:{}".format(i, loss_mse_v))
    xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
    grid = np.c_[xx.ravel(), yy.ravel()] #生成网格数据点配对成为矩阵
    probs = sess.run(y, feed_dict={x: grid})
    probs = probs.reshape(xx.shape)
ax3 = fig.add_subplot(223)
ax3.scatter(X[:,0], X[:,1], c = Y_color)
ax3.contour(xx, yy, probs, levels = [.5])
ax3.set_title("Without Regularizer")
ax4 = fig.add_subplot(224)
ax4.scatter(X[:,0], X[:,1], c = Y_color)
cir1 = Circle(xy = (0.0, 0.0), radius=math.sqrt(2), alpha=0.5)
ax4.add_patch(cir1)
ax4.set_title("Perfect Classifier")
plt.show()






