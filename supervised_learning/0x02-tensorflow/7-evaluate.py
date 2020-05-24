#!/usr/bin/env python3
"""this modual contains the function for task 7"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """ evaluates the output of a nerual network"""
    saver = tf.train.Saver()
    sp = save_path
    with tf.Session() as sess:
        sess = saver.restore(sess, sp)
        print("sess -> ", sess)
        prediction = sess.run(predict, feed_dict={inputdata: X_valid, one_hot: Y_valid})
        print("prediction -> ", prediction)
        # accuracy = sess.run(accuracy, feed_dict={inputdata: X_valid, one_hot: Y_valid})
        # print("accuracy -> ", accuracy)
        # loss = sess.run(cost, feed_dict={inputdata: X_valid, one_hot: Y_valid})
        # print("loss -> ", loss)
