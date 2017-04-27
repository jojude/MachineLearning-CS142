#
#  stocksML.py
#  stocksML
#
#  Created by Jude Joseph on 03/23/17.

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import stocks2 as nn
import pandas
import numpy
import math

#GLOBAL CONSTANTS
scaler = MinMaxScaler(feature_range=(0, 1))
numpy.random.seed(7)
look_back = 1

# LOAD DATASET
# reverse csv file ( order: old to new ) and set
# date/dataset values to float represenation for matrix applications
dataframe = pandas.read_csv('sample.csv', usecols=[1], engine='python', skipfooter=3)
dataframe = dataframe.reindex(index=dataframe.index[::-1])
dataframe.head()
dataset = dataframe.values
dataset = dataset.astype('float32')

# BUILDING TRAINING AND TESTING SETS FROM DATASET
# practicing normalization of data
# training on 70% of data then testing on last 30%
# finally reshape into matrix = [samples, time steps, features]
dataset = nn.normalize(dataset, scaler)
train, test = nn.cross_validate(dataset, 0.7)
trainX, trainY = nn.create_dataset(train, look_back)
testX, testY = nn.create_dataset(test, look_back)

trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# BUILDING A RNN MODEL THROUGH TRAINING
# calls method to build a LSTM neural network to train data
model = nn.RNN_LSTM(trainX, trainY)

# BUILD PREDICTION BASED ON MODEL
train_prediction, test_prediction = nn.predict(trainX, testX, model)

# DENORMALIZE IN ORDER TO MAP ONTO PLOT ACCURATELY
train_Yhat, trainY, test_Yhat, testY = nn.denormalize(train_prediction, test_prediction, trainY, testY, scaler)

# CALCULATE MEAN SQUARED ERROR
# return average range of error between price of predicted stock
# and the real value of the stock on certain date
nn.calculate_L2_error(trainY, train_Yhat, testY, test_Yhat)

# PLOT DATA
# build plot of original dataset,
# initial 70% training and final #30% testing
# in which train and test plots of shifted w.r.t
# original data to visualize error
nn.build_plot(dataset, train_Yhat, test_Yhat, scaler)

#end
