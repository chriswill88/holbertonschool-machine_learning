#!/usr/bin/env python3
"""this modual contains the function for task 7"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """ evaluates the output of a nerual network"""
    sp = save_path

    with tf.Session() as ses:
        saver = tf.train.import_meta_graph(sp + ".meta")
        saver.restore(sess=ses, save_path=sp)
        graph = tf.get_default_graph()

        print("ses -> ", ses)
        print("saver -> ", saver)
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        acc = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        pred = tf.get_collection("y_pred")[0]

        print("prediction -> ", pred)
        print("acc -> ", acc)
        print("los -> {}\n\n\n".format(loss))
        return pred, acc, loss
