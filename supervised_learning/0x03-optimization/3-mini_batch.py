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

    with tf.Session() as ses:
        saver = tf.train.import_meta_graph(lp + ".meta")
        saver.restore(sess=ses, save_path=lp)
        graph = tf.get_default_graph()

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accu = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train = tf.get_collection("train_op")[0]

        acc = ses.run(accu, feed_dict={x: X_train, y: Y_train})
        Vacc = ses.run(accu, feed_dict={x: X_valid, y: Y_valid})
        cost = ses.run(loss, feed_dict={x: X_train, y: Y_train})
        Vcost = ses.run(loss, feed_dict={x: X_valid, y: Y_valid})
        print("After {} epochs:".format(0))
        print("\tTraining Cost: {}".format(cost))
        print("\tTraining Accuracy: {}".format(acc))
        print("\tValidation Cost: {}".format(Vcost))
        print("\tValidation Accuracy: {}".format(Vacc))

        for i in range(epochs):
            X_train, Y_train = shuffle_data(X_train, Y_train)
            step = 0
            for step in range(till_epoch):
                if step != 0 and step % 100 == 0:
                    acc = ses.run(accu, feed_dict={x: X_train, y: Y_train})
                    cost = ses.run(
                        loss, feed_dict={x: X_train, y: Y_train})
                    print("\tStep {}:".format(step))
                    print("\t\tCost: {}".format(cost))
                    print("\t\tAccuracy: {}".format(acc))

                start = step * batch_size
                if batch_size > X_train[start:, :].shape[0]:
                    end = X_train[start:, :].shape[0]
                else:
                    end = start + batch_size
                inp = X_train[start:end, :]
                ypt = Y_train[start:end, :]
                ses.run(train, feed_dict={x: inp, y: ypt})
            acc = ses.run(accu, feed_dict={x: X_train, y: Y_train})
            Vacc = ses.run(accu, feed_dict={x: X_valid, y: Y_valid})
            cost = ses.run(loss, feed_dict={x: X_train, y: Y_train})
            Vcost = ses.run(loss, feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(i + 1))
            print("\tTraining Cost: {}".format(cost))
            print("\tTraining Accuracy: {}".format(acc))
            print("\tValidation Cost: {}".format(Vcost))
            print("\tValidation Accuracy: {}".format(Vacc))

        return saver.save(ses, sp)
