import tensorflow as tf
import numpy as np
import ImgProcesser
import Dataset_processer
def Test():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('D:\\GithubLocalRepo\\Machine-Learning\\Neural_Networks\\My_Model.meta')
        saver.restore(sess, tf.train.latest_checkpoint("D:\\GithubLocalRepo\\Machine-Learning\\Neural_Networks\\"))
        graph = tf.get_default_graph()
        y = tf.get_collection('pred_network')[0]
        x = graph.get_operation_by_name('x').outputs[0]
        #要把y加入到pred_network的collection中
        #设计模型时要写name参数才能找到！！！
        while(1):
            Str = input("input a hand writing picture:\n")
            Vec = Dataset_processer.img2vector(ImgProcesser.picTo01("D:\\GithubLocalRepo\\Machine-Learning\\Neural_Networks\\{}".format(Str)))
            Xtest = Vec.reshape((1, 1024))
            Probs = sess.run(y, feed_dict={x:Xtest})
            ListProbs = Probs[0].tolist()
            print("handwriting is predict to be {}".format(str(ListProbs.index(max(ListProbs)))))

if __name__ == "__main__":
    Test()