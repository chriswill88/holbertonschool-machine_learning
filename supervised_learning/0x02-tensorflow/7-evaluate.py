#!/usr/bin/env python3
"""this modual contains the function for task 7"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """ evaluates the output of a nerual network"""
    sp = save_path
    with tf.Session() as ses:
        saver = tf.train.import_meta_graph(save_path + ".meta")
        ses = saver.restore(sess=ses, save_path=sp)
        print("ses -> ", ses)
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        pred = tf.get_collection("y_pred")[0]

        acc = ses.run(accuracy, feed_dict={"x": X, "y": Y})
        los = ses.run(loss, feed_dict={"x": X, "y": Y})

        print("prediction -> ", pred)
        print("acc -> ", acc)
        print("los -> ", loss)
        return pred, acc, los
