#!/usr/bin/env python3
"""this modual contains the function for task 7"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """ evaluates the output of a nerual network"""
    saver = tf.train.Saver
    sp = save_path
    with tf.Session() as sess:
        sess = saver.restore(sess, save_path=sp)
        print("sess -> ", sess)
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        pred = tf.get_collection("y_pred")[0]

        pre = sess.run(pred, feed_dict={"x": X, "y": Y})
        acc = sess.run(accuracy, feed_dict={"x": X, "y": Y})
        los = sess.run(loss, feed_dict={"x": X, "y": Y})

        print("prediction -> ", pred)
        print("acc -> ", acc)
        print("los -> ", loss)
        return pred, acc, los
