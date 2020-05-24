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
    layer_sizes is a list containing the number of nodes in each layer of the network
    activations is a list containing the activation functions for each layer of the network
    alpha is the learning rate
    iterations is the number of iterations to train over
    save_path designates where to save the model
    """
    nx = X_train.shape[1]
    classes = Y_train.shape[1]
    vnx, vclass = X_valid.shape

    print("X -> ", X_train.shape, "\nY -> ", Y_train.shape)

    inputdata, one_hot = create_placeholders(nx, classes)
    print(inputdata, one_hot)
    predict = forward_prop(inputdata, layer_sizes, activations)
    print(predict)
    cost = calculate_loss(one_hot, predict)
    print(cost)

    accuracy = calculate_accuracy(one_hot, predict)
    train = create_train_op(cost, alpha)

    print(inputdata, "\n", predict, "\n", accuracy, "\n", cost, "\n", train)
    ini = tf.global_variables_initializer()

    saver = tf.train.Saver
    sess = tf.Session()
    sess.run(ini)

    for i in range(iterations):
        print(sess.run(train, feed_dict={inputdata: X_train, one_hot: Y_train}))

        if i % iterations == 0 or i == iterations - 1:
            print("i = {}\n", i)
            print("\tTraining Cost: {}".format(cost))
            print("\tTraining Accuracy: {}".format(accuracy))
            print("\tValidation Cost: {}".format(cost))
            print("\tValidation Accuracy: {}".format(accuracy))
