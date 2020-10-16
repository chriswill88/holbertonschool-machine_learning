import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


class PreprocessData:
    """This class contains tools that are used to preprocess the data"""
    def __init__(self, data):
        """
        @data is the raw dataset
        """
        self.raw_data = data

        self.train_data, self.test_data, self.scalar\
            = self.preprocess_data(data)

        self.tf_train = tf.data.Dataset.from_tensor_slices(
            self.train_data)

        self.tf_test = tf.data.Dataset.from_tensor_slices(self.test_data)

    def preprocess_data(self, bit):
        """
        preprovess_data - this function preprocesses the dataset:

        1. it truncats the dataset to just a single column
        2. it also limits the data to show a point every hour
        3. it scales the data between 0 and 1
        4. it parses the data into a training set and test set
        """
        # hourly data, removeing nulls, only close column
        bit = bit[::60]
        bit = bit.where(pd.notna(bit), bit.mean(), axis='columns')
        bit = bit['Close']

        # scaling data 0, 1
        scalar = MinMaxScaler(feature_range=(0, 1))
        scaledBit = scalar.fit_transform(np.array(bit).reshape(-1, 1))

        # Train/validation/test ordered by time
        train_size = int(len(scaledBit)*.7)

        # parsing datasets
        train_data, test_data = scaledBit[:train_size], scaledBit[train_size:]

        # 24 hour data set: 1 hour after y values
        trainX, trainY = self.dataset(24, train_data)
        testX, testY = self.dataset(24, test_data)
        return (trainX, trainY), (testX, testY), scalar

    def dataset(self, timestep, data):
        """this function parse the data by certain timesteps"""
        dataX, dataY = [], []
        for i in range(len(data) - timestep):
            a = data[i:i+timestep]
            dataX.append(a)
            dataY.append(data[i + timestep])
        return np.array(dataX), np.array(dataY)

    def unprocess_data(self, data):
        """this unprocesses the data"""
        scalar = self.scalar
        return scalar.inverse_transform(data)
