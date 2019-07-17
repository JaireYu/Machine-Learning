import tensorflow as tf
import forward
import Dataset_processer
BATCH_SIZE = 200
STEPS = 30000
def backward(X, Y_, trainsize, Xtest):
    x = tf.placeholder(tf.float32, shape=(None, 1024), name="x")
    y_ = tf.placeholder(tf.float32, shape=(None, 10), name="y_") #数据集的实际输出
    y = forward.forward(x, 0.01)
    global_step = tf.Variable(0, trainable=False)
    loss = tf.reduce_mean(tf.square(y-y_)) + tf.add_n(tf.get_collection("losses"))
    learning_rate = tf.train.exponential_decay(0.001, global_step, LEARNING_RATE_STEP, LEARNING_RATE_DECAY, staircase=False)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    tf.add_to_collection('pred_network', y)
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(STEPS):
            start = (i * BATCH_SIZE) % trainsize
            end = start + BATCH_SIZE
            sess.run(train_step, feed_dict={x:X[start: end], y_:Y_[start: end]})
            if i % 10000 == 0:
                loss_mse_v = sess.run(loss, feed_dict={x:X, y_:Y_})
                print("STEP:{}, LOSS:{}".format(i, loss_mse_v))
        saver = tf.train.Saver()
        saver.save(sess, 'D:\\GithubLocalRepo\\Machine-Learning\\Neural_Networks\\My_Model')
        Probs = sess.run(y, feed_dict={x: Xtest})
        return Probs

if __name__ == "__main__":
    X, Y_, trainsize = Dataset_processer.GetDataSet()
    Xtest, Y_test, testsize = Dataset_processer.GetTestDataSet()
    LEARNING_RATE_STEP = trainsize / BATCH_SIZE
    LEARNING_RATE_DECAY = 0.99
    Probs = backward(X, Y_, trainsize, Xtest)
    PredictLabels = [prob.tolist().index(max(prob.tolist())) for prob in Probs]
    TestDataLabels = [prob.tolist().index(1) for prob in Y_test]
    error_sample = 0
    for i in range(testsize):
        if(PredictLabels[i] != TestDataLabels[i]):
            error_sample += 1
    print("error_rate = {}".format(float(error_sample)/testsize))
