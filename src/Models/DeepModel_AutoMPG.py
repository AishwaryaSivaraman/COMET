#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

sys.path.append('./src/')

def make_data(configurations):
    column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                    'Acceleration', 'Model Year', 'Origin']
    train_dataset = []
    train_labels = []
    test_dataset= []
    test_labels= []
    
    train_path = configurations['model_dir'] + '/train_data.csv'
    test_path = configurations['model_dir'] + '/test_data.csv'
    if (os.path.isfile(train_path)):
        trainX = pd.read_csv(train_path, index_col=0)
        testX = pd.read_csv(test_path, index_col=0)
    else:
        print("Please provide path to test and train csv files")
        sys.exit(0)
    
    trainY = trainX.pop('MPG')
    testY = testX.pop('MPG')
    
    train_dataset.append(trainX)
    test_dataset.append(testX)

    train_labels.append(trainY)
    test_labels.append(testY)
    min_max_dict = getMinMaxRangeOfFeatures(trainX, configurations['feature_names'])
    return train_dataset, train_labels, test_dataset, test_labels, min_max_dict

def evaluate(model, test_dataset, test_labels, isClassification):
    scores = model.evaluate(test_dataset, test_labels, verbose=0)
    if isClassification:
        return scores[1]
    else:
        return scores[0]

def update_batch (model, layer_size, batch_data, batch_label, data_dir, batch_size):
    history=model.fit(batch_data, batch_label, epochs=1, batch_size=batch_size, validation_split = 0.2, verbose=0)
    for i in range(layer_size):
        weight = model.layers[i].get_weights()[0]
        bias = model.layers[i].get_weights()[1]
        np.savetxt(data_dir+"/weights_layer%d.csv"%(i),weight,delimiter=",")
        np.savetxt(data_dir+"/bias_layer%d.csv"%(i),bias,delimiter=",")        
    return model

def output(model, datapoint):
    x_point = pd.DataFrame(datapoint)
    return model.predict(x_point.transpose())


def getMinMaxRangeOfFeatures(dataset, column_names):
    min_max = {}
    for i in column_names:
        index = column_names.index(i)
        min_max[i] = [min(dataset[column_names[index]]),max(dataset[column_names[index]])]
    return min_max