#!/usr/bin/env python3
"""this module contains the PreprocessData class"""
import numpy as np
import tensorflow as tf


class PreprocessData:
    """This class contains tools that are used to preprocess the data"""
    def __init__(self):
        """
        @data is the raw dataset
        """
        data = np.genfromtxt(
            'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', delimiter=',')
        self.raw_data = data
        # print(data.shape)
        # hourly data, removeing nulls, only close column
        self.cdata = data[:, 1]  # close column
        # print(self.cdata.shape)
        self.tdata = self.cdata[::60]  # hour interval data
        # print(self.tdata.shape)

        self.sdata = self.moving_average(self.tdata, 20)[:, np.newaxis]
        # smooth over using moving ave
        # print("sdata", self.sdata.shape)
        # mean and std for normalization
        self.mean = None
        self.std = None

        # train/valid/test
        self.train_data, self.valid_data, self.test_data \
            = self.preprocess_data(self.sdata)

        # scaled sdata
        self.scaled_sdata = self.scale_data(self.sdata)

        # tensor dataset
        self.tf_train = tf.data.Dataset.from_tensor_slices(
            self.train_data).batch(32)
        self.tf_valid = tf.data.Dataset.from_tensor_slices(
            self.valid_data).batch(32)
        self.tf_test = tf.data.Dataset.from_tensor_slices(
            self.test_data).batch(32)

    def preprocess_data(self, sdata):
        """
        preprocess_data - this function preprocesses the dataset:
        @sdata - moveing ave data
        """
        n = len(sdata)
        # parsing datasets
        train_data, valid_data, test_data = sdata[:int(n*.7)], \
            sdata[int(n*.7):int(n*.9)], sdata[int(n*.9):]

        # setting mean and std for normalization
        self.mean = np.mean(train_data)
        self.std = np.std(train_data)

        # normalize data
        train_data = self.scale_data(train_data)
        valid_data = self.scale_data(valid_data)
        test_data = self.scale_data(test_data)

        # 24 hour data set: 1 hour after y values
        trainX, trainY = self.dataset(24, train_data)
        validX, validY = self.dataset(24, valid_data)
        testX, testY = self.dataset(24, test_data)

        # print(trainX.shape, trainY.shape)
        return (trainX, trainY), (validX, validY), (testX, testY)

    def moving_average(self, a, n=3):
        """
        this function reformates the data based on
        moving average to fill in all nulls
        """
        ret = np.nancumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    def dataset(self, timestep, data):
        """this function parse the data by certain timesteps"""
        dataX, dataY = [], []
        for i in range(len(data) - timestep):
            a = data[i:i+timestep]
            dataX.append(a)
            dataY.append(data[i + timestep])
        return np.array(dataX), np.array(dataY)

    def scale_data(self, data):
        """this function scales out the data"""
        return (data - self.mean)/self.std

    def unscale_data(self, data):
        """this unprocesses the data"""
        return (data + self.mean)*self.std
