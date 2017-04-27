#
#  stocks2.py
#  stocks2
#
#  Created by Jude Joseph on 03/23/17.
#
#  Thanks to : Jason Brownlee
#  http: //machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-pythonkeras/
#

import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error

#constant
look_back = 1

# DEF: NORMALIZE
# LSTM is sensitive to the scale of input data
# so we will transform it to fit the model
def normalize(dataset, scaler):
    dataset = scaler.fit_transform(dataset)
    return dataset

# DEF: CROSS_VALIDATE
# x represents a percentile of which the dataset
# will be split into test and train sets
def cross_validate(dataset, x):
    size_trainingSet = int(len(dataset) * x)
    size_testingSet = len(dataset) - size_trainingSet
    train, test = dataset[0:size_trainingSet,:], dataset[size_trainingSet:len(dataset),:]
    return train, test

# DEF: CREATE_DATASET
# separate data into features and targets
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

# DEF: SPLITDATASET
# addition helper function to make
# training and testing features and
# targets into respectively sized matrices
def splitdataset(train, test):
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    return trainX, trainY, testX, testY

# DEF: RNN_LSTM
# this is where most of the work happens using keras
# we build a model with 4 LMST cell bodies within
# one recurrent neural network with relu activation
# then apply gradient descent with the adam optimizer
# analogous to TensorFlow methods then fit then model
def RNN_LSTM(trainX, trainY):
    model = Sequential()
    model.add(LSTM(4, input_dim=look_back))
    model.add(Dense(1,init='uniform',activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
    model.fit(trainX, trainY, nb_epoch=100, batch_size=10, verbose=2)

    return model

# DEF: PREDICT
# helper function to apply predictions on
# training and testing data
def predict(trainX, testX, model):
    train_prediction = model.predict(trainX)
    test_prediction = model.predict(testX)
    return train_prediction, test_prediction

# DEF: DENORMALIZE
# invert predictions // denormalize in order to
# plot correctly on graph
def denormalize(train_yhat, test_yhat, trainY, testY, scaler):
    train_Yhat = scaler.inverse_transform(train_yhat)
    trainY = scaler.inverse_transform([trainY])
    test_Yhat = scaler.inverse_transform(test_yhat)
    testY = scaler.inverse_transform([testY])

    return train_Yhat, trainY, test_Yhat, testY

# DEF: CALCULATE_L2_ERROR
# calculates mean squared error on both
# training and testing sets and prints to
# terminal. Notice that we are using sqrt
# so we can get the average difference of
# stock prices predicated and the real prices
# the lower the average the more accurate the model
def calculate_L2_error(trainY, train_Yhat, testY, test_Yhat):
    # calculate root mean squared error
    err_train = math.sqrt(mean_squared_error(trainY[0], train_Yhat[:,0]))
    err_test = math.sqrt(mean_squared_error(testY[0], test_Yhat[:,0]))
    print('Training Score: %.2f RMSE' % (err_test))
    print('Testing Score: %.2f RMSE' % (err_train))

# DEF: BUILD_PLOT
# helper function used to build data plot
# to visualize predication compared to the
# real data
def build_plot(dataset, train_Yhat, test_Yhat, scaler):
    #un-invert 'whole' original dataset
    dataset_plot = scaler.inverse_transform(dataset)
    
    # move training and testing lots a bit lower than original data
    training_plot = numpy.empty_like(dataset)
    training_plot[:, :] = numpy.nan
    training_plot[look_back:len(train_Yhat)+look_back, :] = train_Yhat

    testing_plot = numpy.empty_like(dataset)
    testing_plot[:, :] = numpy.nan
    testing_plot[len(train_Yhat)+(look_back*2)+1:len(dataset)-1, :] = test_Yhat

    # plot original data and predictions
    fig, ax = plt.subplots()
    fig.suptitle('NASDAQ:GOOG', fontsize=14, fontweight='bold')
    ax.plot(dataset_plot, 'b', label='True Data')
    ax.plot(training_plot, 'g', label='Train Predication')
    ax.plot(testing_plot, 'r', label='Test Prediction')
    
    #label x and y axis
    ax.set_ylabel('Stock Prices($)')
    ax.set_xlabel('2016-Mar-28 to 2017-Mar-23')
    
    # add legend
    legend = ax.legend(loc='upper left', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    
    for label in legend.get_texts():
        label.set_fontsize('medium')

    for label in legend.get_lines():
        label.set_linewidth(1.0)
    
    plt.show()

#end
