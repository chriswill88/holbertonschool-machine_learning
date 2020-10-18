#!/usr/bin/env python3
"""this module will be used to forcast_btc"""
import numpy as np
# import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
pp = __import__('preprocess_data').PreprocessData

"""
Your model should use the past 24 hours of BTC data to predict
the value of BTC at the close of the following hour
(approximately how long the average transaction takes):

The datasets are formatted such that every row represents
a 60 second time window containing:
"""

# # open and view csv file

# bit = np.genfromtxt(
#     'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', delimiter=',')

# bit = pd.read_csv('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv')
# bit = pd.read_csv('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv')
data = pp(bit)

trainX, trainY = data.train_data
# print(trainX.shape)
validX, validY = data.valid_data
testX, testY = data.test_data

train_tf = data.tf_train
valid_tf = data.tf_valid
test_tf = data.tf_test

# print("tx shape", trainX.shape)
# Model
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(1, input_shape=(24, 1)))
model.add(tf.keras.layers.Dense(1))
model.compile(loss='mse', optimizer=Adam(lr=.00007), metrics=['accuracy'])
model.summary()

# print("before train", train_tf)
history = model.fit(
    trainX, trainY, validation_data=(
        validX, validY), batch_size=64, epochs=20).history
model.save('forecast.h5')
# print("Saved model")
# print(history)

# history
acc = history['acc']
val_acc = history['val_acc']
loss = history['loss']
val_loss = history['val_loss']

# Prediction

# train_pred = model.predict(trainX)
# test_pred = model.predict(testX)
# train_pred = []
# for i in range(len(trainX)):
#     train_pred.append(model.predict(trainX[np.newaxis, i]))

# test_pred = []
# for i in range(len(testX)):
#     test_pred.append(model.predict(testX[np.newaxis, i]))

train_pred = np.squeeze(np.array(train_pred), 1)
# print(train_pred.shape)
test_pred = np.squeeze(np.array(test_pred), 1)
# print(test_pred.shape)
# train_pred = data.unprocess_data(train_pred)
# test_pred = data.unprocess_data(test_pred)
# testY = data.unprocess_data(testY)

# plt.figure(1)
# plt.title("train loss")
# plt.plot(loss, label="train")
# plt.plot(val_loss, label="val")

# plt.figure(3)
# plt.title("train")
# plt.plot(trainY, label="trainY")
# plt.plot(test_pred,  label="train_pred")

# plt.figure(4)
# plt.title("test")
# plt.plot(test_pred, label='test_pred', )
# plt.plot(train_pred[:, 0], label="test")

# plt.figure(5)
# plt.title("pred comp")
# plt.plot(data.scaled_sdata, label="data")
# # plt.plot(train_pred, label="test")
# # plt.plot(test_pred,  label="train")
# plt.legend()

# plt.show()
