#!/usr/bin/env python3
"""This modual contains the function for task 3"""
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(
    X_train, Y_train, X_valid, Y_valid,
    batch_size=32, epochs=5,
        load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):
    """
        X_train is a numpy.ndarray of shape (m, 784) containing the training data
        m is the number of data points
        784 is the number of input features
        Y_train is a one-hot numpy.ndarray of shape (m, 10) containing the training labels
        10 is the number of classes the model should classify
        X_valid is a numpy.ndarray of shape (m, 784) containing the validation data
        Y_valid is a one-hot numpy.ndarray of shape (m, 10) containing the validation labels
        batch_size is the number of data points in a batch
        epochs is the number of times the training should pass through the whole dataset
        load_path is the path from which to load the model
        save_path is the path to where the model should be saved after training
        Returns: the path where the model was saved
    """
    lp, sp = load_path, save_path
    m, ipF = X_train.shape
    till_epoch = int(ipF / batch_size) + (ipF % batch_size > 0)
    step = 0
    bs = batch_size

    with tf.Session() as ses:
        print("in the ses")
        saver = tf.train.import_meta_graph(lp + ".meta")
        print("saver -> ", saver)
        saver.restore(sess=ses, save_path=lp)
        graph = tf.get_default_graph()
        print("ses -> ", ses)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accu = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        pred = tf.get_collection("y_pred")[0]
        train = tf.get_collection("train_op")[0]

        print(x, y, accu, loss)

    for i in range(epochs):
        X_train, Y_train = shuffle_data(X_train, Y_train)
        step = 0
        acc = ses.run(accu, feed_dict={x: X_train, y: Y_train})
        Vacc = ses.run(acc, feed_dict={x: X_valid, y: Y_valid})
        cost = ses.run(loss, feed_dict={x: X_train, y: Y_train})
        Vcost = ses.run(loss, feed_dict={x: X_valid, y: Y_valid})
        print("After {} epochs:".format(i))
        print("\tTraining Cost: {}".format(cost))
        print("\tTraining Accuracy: {}".format(acc))
        print("\tValidation Cost: {}".format(Vcost))
        print("\tValidation Accuracy: {}".format(Vacc))

        for x in range(till_epoch):  # use vectorizing
            if step != 0 and step % 100 == 0:
                acc = ses.run(accu, feed_dict={x: X_train, y: Y_train})
                cost = ses.run(loss, feed_dict={x: X_train, y: Y_train})
                print("\tStep {}:".format(step))
                print("\t\tCost: {}".format(cost))
                print("\t\tAccuracy: {}".format(acc))

            start = step * batch_size
            if batch_size > X_train[:, start:].shape[1]:
                end = X_train[:, start:].shape[1]

            inp = X_train[:, start:end]
            ypt = Y_train[:, start:end]

            ses.run(train, feed_dict={x: inp, y: ypt})
            step += 1
    return saver.save(ses, sp)
