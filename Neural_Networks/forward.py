import tensorflow as tf
def forward(x, regularizer):
    w1 = get_weight([1024,100], regularizer, "w1")
    b1 = get_bias([100], "b1")
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    w2 = get_weight([100, 10], regularizer, "w2")
    b2 = get_bias([10], "b2")
    y = tf.matmul(y1, w2) + b2
    return y

def get_weight(shape, regularizer, name):
    w = tf.Variable(tf.random_normal(shape), dtype=tf.float32, name=name)#使用正态分布随机生成w
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape, name):
    b = tf.Variable(tf.constant(0.01, shape=shape), name=name)
    return b

