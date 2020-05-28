#!/usr/bin/env python3
"""This modual contains the function for task 3"""
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(
    X_train, Y_train, X_valid, Y_valid,
    batch_size=32, epochs=5,
        load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):
    """
        X_train is a ndarray of shape (m, 784) containing training data
        m is the number of data points
        784 is the number of input features
        Y_train is a one-hot ndarray of (m, 10) containing the training labels
        10 is the number of classes the model should classify
        X_valid is a ndarray of (m, 784) containing the validation data
        Y_valid is a one-hot ndarray of (m, 10) containing/ validation labels
        batch_size is the number of data points in a batch
        epochs is the number of times the training passes through the dataset
        load_path is the path from which to load the model
        save_path is the path to where the model should be saved after training
        Returns: the path where the model was saved
    """
    lp, sp = load_path, save_path
    m, ipF = X_train.shape
    till_epoch = int(m / batch_size) + (m % batch_size > 0)
    step = 0
    bs = batch_size
    X_shuf, Y_shuf = X_train, Y_train

    with tf.Session() as ses:
        saver = tf.train.import_meta_graph(lp + ".meta")
        saver.restore(ses, lp)
        graph = tf.get_default_graph()

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accu = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train = tf.get_collection("train_op")[0]

        for i in range(epochs + 1):
            step = 0
            acc = ses.run(accu, feed_dict={x: X_train, y: Y_train})
            Vacc = ses.run(accu, feed_dict={x: X_valid, y: Y_valid})
            cost = ses.run(loss, feed_dict={x: X_train, y: Y_train})
            Vcost = ses.run(loss, feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(cost))
            print("\tTraining Accuracy: {}".format(acc))
            print("\tValidation Cost: {}".format(Vcost))
            print("\tValidation Accuracy: {}".format(Vacc))
            if i < epochs:
                X_shuf, Y_shuf = shuffle_data(X_train, Y_train)
                for step in range(till_epoch):
                    start = step * batch_size
                    if batch_size > X_shuf[start:, :].shape[0]:
                        end = start + X_shuf[start:, :].shape[0]
                    else:
                        end = start + batch_size
                    inp = X_shuf[start:end, :]
                    ypt = Y_shuf[start:end, :]

                    if step != 0 and (step + 1) % 100 == 0:
                        acc = ses.run(accu, feed_dict={x: inp, y: ypt})
                        cost = ses.run(
                            loss, feed_dict={x: inp, y: ypt})
                        print("\tStep {}:".format(step + 1))
                        print("\t\tCost: {}".format(cost))
                        print("\t\tAccuracy: {}".format(acc))
                    ses.run(train, feed_dict={x: inp, y: ypt})
        return saver.save(ses, sp)
