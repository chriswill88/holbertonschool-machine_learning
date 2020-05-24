#!/usr/bin/env python3
"""this modual contains the function for task 6"""
import tensorflow as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(
        X_train, Y_train, X_valid, Y_valid,
        layer_sizes, activations,
        alpha, iterations,
        save_path="/tmp/model.ckpt"):

    """
    X_train is a numpy.ndarray containing the training input data
    Y_train is a numpy.ndarray containing the training labels
    X_valid is a numpy.ndarray containing the validation input data
    Y_valid is a numpy.ndarray containing the validation labels
    layer_sizes contains the number of nodes in each layer of the network
    activations contains the activation functions for each layer of the network
    alpha is the learning rate
    iterations is the number of iterations to train over
    save_path designates where to save the model
    """
    sp = save_path
    nx = X_train.shape[1]
    classes = Y_train.shape[1]
    vnx, vclass = X_valid.shape

    inputdata, one_hot = create_placeholders(nx, classes)
    predict = forward_prop(inputdata, layer_sizes, activations)
    cost = calculate_loss(one_hot, predict)

    accuracy = calculate_accuracy(one_hot, predict)
    train = create_train_op(cost, alpha)
    ini = tf.global_variables_initializer()

    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(ini)

    tf.add_to_collection(name="x", value=inputdata)
    tf.add_to_collection(name="y", value=one_hot)
    tf.add_to_collection(name="y_pred", value=predict)
    tf.add_to_collection(name="loss", value=cost)
    tf.add_to_collection(name="accuracy", value=accuracy)
    tf.add_to_collection(name="train_op", value=train)

    for i in range(iterations):
        sess.run(predict, feed_dict={inputdata: X_train, one_hot: Y_train})
        if i % 100 == 0 or i == iterations:
            print("After {} iterations:".format(i))
            print("\tTraining Cost: {}".format(
                sess.run(
                    cost,
                    feed_dict={inputdata: X_train, one_hot: Y_train})))
            print("\tTraining Accuracy: {}".format(
                sess.run(
                    accuracy,
                    feed_dict={inputdata: X_train, one_hot: Y_train})))
            print("\tValidation Cost: {}".format(
                sess.run(
                    cost,
                    feed_dict={inputdata: X_valid, one_hot: Y_valid})))
            print("\tValidation Accuracy: {}".format(
                sess.run(
                    accuracy,
                    feed_dict={inputdata: X_valid, one_hot: Y_valid})))
        if i != iterations:
            sess.run(train, feed_dict={inputdata: X_train, one_hot: Y_train})
    return saver.save(sess, sp)
