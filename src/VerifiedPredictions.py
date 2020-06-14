import os
import sys
import ast
import time
import pickle
import argparse
import importlib
import numpy as np
import pandas as pd
import tensorflow as tf

from Envelope import getEnvelopeResult, generate_counter_example
from Utils import makeDir,write_to_csv,copyfiles,copyFile, readConfigurations
from ModelCalls import generate_data, make_batch, evaluate_model, update_model


def getEnvelopePredictions():
    column_names = configurations['column_names']
    configurations['weight_files'] = os.path.dirname(model_file)+"/"
    predictionsDir = os.path.dirname(test_file) + '/'
    raw_dataset = pd.read_csv(test_file, 
                          na_values = "?", comment='\t',
                          sep=",", skipinitialspace=True)
    dataset = raw_dataset.copy()
    dataset.tail()
    dataset = dataset.dropna()
    dataset = dataset.drop(columns=[col for col in dataset if col not in column_names])
    dataset.drop(dataset.index[:1], inplace=True)
    model_file_h5 = model_file + "model.h5"
    model = tf.keras.models.load_model(model_file_h5)
    predictions = []
    upper_envelope_predictions = []
    lower_envelope_predictions = []
    upper_cg_count = 0
    lower_cg_count = 0
    for index, row in dataset.iterrows():
      data_point = row
      f_x = output(model, data_point)[0][0]
      counter_example, elapsed_time, vio, data_index = generate_counter_example(configurations, counter_example_generator_upper, data_point, 0, 0, f_x, 0)

      counter_example_lower, elapsed_time, vio, data_index = generate_counter_example(configurations, counter_example_generator_lower, data_point, 0, 0, f_x, 0)
      predictions.append(f_x)
      if counter_example is not None:
        upper_envelope_predictions.append(output(model, counter_example)[0][0])
        upper_cg_count = upper_cg_count + 1
      else:
        upper_envelope_predictions.append(f_x)
      if counter_example_lower is not None:
        lower_envelope_predictions.append(output(model, counter_example_lower)[0][0])
        lower_cg_count = lower_cg_count + 1
      else:
        lower_envelope_predictions.append(f_x)
    np.savetxt(predictionsDir + "monotonic_predictions.csv",(np.array([predictions,upper_envelope_predictions,lower_envelope_predictions])).T, delimiter = ',',header= "f_x, upper envelope, lower envelope")
    print("Monotonic predictions are saved here: " + str(predictionsDir + "monotonic_predictions.csv"))
    return predictions, upper_envelope_predictions, lower_envelope_predictions

parser = argparse.ArgumentParser(description='Envelope Prediction')
parser.add_argument('config_file', metavar='c', type=str,
                    help='configuration file')
parser.add_argument('test_file', metavar='f', type=str,
                    help='test file')

args = parser.parse_args()
config_file = args.config_file
test_file = args.test_file
configurations = readConfigurations(config_file)
model_file = configurations['model_dir']
solver_times = []

#--- import the nn model --------
sys.path.append('./src/Models')
evaluate = importlib.__import__(configurations['model']).evaluate
output = importlib.__import__(configurations['model']).output

#----- import solver functions --------
counter_example_generator_upper = importlib.__import__(configurations['solver']).counter_example_generator_upper_env
counter_example_generator_lower = importlib.__import__(configurations['solver']).counter_example_generator_lower_env

counter_pair_generator = importlib.__import__(configurations['solver']).counter_pair_generator

#-------- COMET ---------
getEnvelopePredictions()