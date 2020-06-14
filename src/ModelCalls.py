
import sys
import math
import time
import random
import pickle
import logging
import importlib
import statistics
import numpy as np
import pandas as pd
import multiprocessing

from typing import List
from ast import literal_eval
from datetime import datetime
from joblib import Parallel, delayed
from Envelope import getEnvelopeResult
from Utils import makeDir,write_to_csv,copyfiles,copyFile


#generate train and test sets
def generate_data(make_cv_data, configurations):
    train_data, train_labels, test_data, test_labels,min_max_dict = make_cv_data(configurations)
    return train_data, train_labels, test_data, test_labels,min_max_dict

#given a size, randomly assing index of trainset to a batch
def make_batch(train_data, train_labels, configurations):
    data_size = train_data.shape
    batch_size = configurations['model_parameters']['batch_size']
    batches = []
    batches_labels = []
    data_indexes = list(range(0, data_size[0]))
    start = 0
    while (start+batch_size)<data_size[0]:
        batch= data_indexes[start:start+batch_size]
        batches.append(batch)
        start += batch_size
    batch = data_indexes[start:]
    batches.append(batch)
    return batches

def evaluate_model(NN_model, test_data, test_labels, configurations, evaluate):
    test_dataframe = pd.DataFrame(test_data)
    label_dataframe = pd.DataFrame(test_labels)
    score = evaluate(NN_model, test_dataframe, label_dataframe, configurations['is_classification'])
    return score

#fit the model based on the new counter batch and return a new model
def update_model(counter_batch, batch_labels, NN_model, configurations, update_batch):
    weight_dir = configurations['weight_files']
    NN_model = update_batch(NN_model,configurations['model_parameters']['layer_size'] ,counter_batch, batch_labels,weight_dir,counter_batch.shape[0])
    return NN_model

