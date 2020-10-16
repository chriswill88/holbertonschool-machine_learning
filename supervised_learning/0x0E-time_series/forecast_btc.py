#!/usr/bin/env python3
"""this module will be used to forcast_btc"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras as K
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
pp = __import__('preprocess_data').PreprocessData

"""
Your model should use the past 24 hours of BTC data to predict
the value of BTC at the close of the following hour
(approximately how long the average transaction takes):

The datasets are formatted such that every row represents
a 60 second time window containing:
"""

# open and view csv file
bit = pd.read_csv('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv')
data = pp(bit)

trainX, trainY = data.train_data
testX, testY = data.test_data

# train_tf = data.tf_train
# test_tf = data.tf_test


# Model
model = K.Sequential()
model.add(K.layers.LSTM(1, return_sequences=True, input_shape=(24, 1)))
model.add(K.layers.LSTM(1, return_sequences=True))
model.add(K.layers.LSTM(1))
model.add(K.layers.Dense(1))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(trainX, trainY, epochs=20, batch_size=64, validation_split=.2)
model.save('forecast.h5')

# prediction
train_pred = model.predict(trainX)
test_pred = model.predict(testX)

train_pred = data.unprocess_data(train_pred)
test_pred = data.unprocess_data(test_pred)
testY = data.unprocess_data(testY)

for i in range(len(testY)):
    print(testY[i], "vs", test_pred[i])
