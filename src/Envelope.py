import sys
import math
import time
import pickle
import logging
import importlib
import statistics
import numpy as np
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt

from typing import List
from ast import literal_eval
from datetime import datetime
from joblib import Parallel, delayed
from Utils import makeDir,write_to_csv,copyfiles,copyFile

def getEnvelopeResult(data, labels, NN_model, monotonic_index, fold, isParallel, counter_example_generator, output, configurations, direction):
    if configurations["scalability"]:
        return getEnvelopeResult_scalability(data, labels, NN_model, monotonic_index, fold,counter_example_generator, output, configurations, direction)
    else:
        if isParallel:
            try:
                return getEnvelopeResult_parallel(data, labels, NN_model, monotonic_index, fold, counter_example_generator, output, configurations, direction)
            except Exception as e:
                print(e)
                print("Parallel processing failed, trying sequential")
                return getEnvelopeResult_seq(data, labels, NN_model, monotonic_index, fold,counter_example_generator, output, configurations, direction)
        else:
            return getEnvelopeResult_seq(data, labels, NN_model, monotonic_index, fold,counter_example_generator, output, configurations, direction)

def getEnvelopeMetrics(results, data, labels, output, NN_model, monotonic_index, direction, isClassification):
    label_count = 0
    no_of_counter = 0
    error = 0.0
    sum_of_violation = 0.0
    violations = []
    for res in results:
        counter_example, elapsed_time, vio, index = res
        point = data.loc[index].values
        f_x = output(NN_model, point)[0][0]
        if counter_example is None:
            #calc mse with f_x
            if isClassification:
                f_x = NN_model.predict_classes(pd.DataFrame(point).transpose())[0][0]
                error = error + abs(f_x-labels.loc[index])
            else:
                error = error + (f_x-labels.loc[index]) * (f_x-labels.loc[index])
        else:
            # f_y = output(NN_model,pd.DataFrame(counter_example))[0][0]
            if not isClassification:
                # f_y = vio
                f_y = output(NN_model,pd.DataFrame(counter_example))[0][0]
                error = error + (f_y-labels.loc[index]) * (f_y-labels.loc[index])
                counter_y = counter_example[monotonic_index]
                if not np.isclose(abs(f_x-f_y),vio,atol=1e-3) or getViolation(f_x, f_y, direction) < 0:
                    print(counter_example)
                    print("f_y from solver "+str(vio))
                    print("The x is " + str(point[monotonic_index]))
                    print("The x' is "+ str(counter_y))
                    print("f(x) is "+ str(f_x))
                    print("f(x') is "+ str(f_y))
                    print("The violation is "+str(getViolation(f_x, f_y, direction)))
                    print("not close enough or negative violation")
                    sys.exit(0)
                sum_of_violation = sum_of_violation + getViolation(f_x, f_y, direction)
                violations = violations + [getViolation(f_x, f_y, direction)]
                no_of_counter = no_of_counter + 1
            else:
                f_y = NN_model.predict_classes(pd.DataFrame(counter_example).transpose())[0][0]
                f_y_prob = output(NN_model,pd.DataFrame(counter_example))[0][0]
                error = error + abs(f_y-labels.loc[index])
                counter_y = counter_example[monotonic_index]

                sum_of_violation = sum_of_violation + abs(f_x-f_y_prob)
                violations = violations + [abs(f_x-f_y_prob)]
                no_of_counter = no_of_counter + 1
        label_count = label_count + 1
    
    avg = 0.0
    max_vio = 0.0
    if len(violations) > 0:
        max_vio = max(violations)
    if no_of_counter > 0 :
        avg = sum_of_violation/no_of_counter*1.0
    if isClassification:
        return (len(data)-error)/len(data),no_of_counter,avg,max_vio
    else:
        return error/len(data), no_of_counter, avg, max_vio

def getEnvelopeResult_seq(data, labels, NN_model, monotonic_index, fold, counter_example_generator, output, configurations, direction):
    dataframe = pd.DataFrame(data)
    label_dataframe = pd.DataFrame(labels)
    all_results = []
    for index,point in dataframe.iterrows():
        f_x = output(NN_model, point)[0][0]
        all_results.append(generate_counter_example(configurations, counter_example_generator, point.copy(), monotonic_index, index, f_x, fold))
    return getEnvelopeMetrics(all_results, data, labels, output, NN_model, monotonic_index, direction,configurations['is_classification'])

def getEnvelopeResult_scalability(data, labels, NN_model, monotonic_index, fold, counter_example_generator, output, configurations, direction):
    dataframe = pd.DataFrame(data)
    label_dataframe = pd.DataFrame(labels)
    all_results_envelope = []
    all_results_prediction = []
    label_count = 0
    no_of_counter = 0
    error = 0.0
    sum_of_violation = 0.0
    violations = []
    isClassification = configurations['is_classification']
    for index,point in dataframe.iterrows():
        start_env_time = datetime.now()
        f_x = output(NN_model, point)[0][0]
        counter_example, elapsed_time, vio, data_index = generate_counter_example(configurations, counter_example_generator, point.copy(), monotonic_index, index, f_x, fold)
        elapse_env_time = datetime.now() - start_env_time
        all_results_envelope.append(elapse_env_time.total_seconds())
        point = data.loc[index].values
        start_env_time = datetime.now()
        f_x = output(NN_model, point)[0][0]
        elapse_env_time = datetime.now() - start_env_time
        all_results_prediction.append(elapse_env_time.total_seconds())
        if counter_example is None:
            #calc mse with f_x
             if isClassification:
                f_x = NN_model.predict_classes(pd.DataFrame(point).transpose())[0][0]
                error = error + abs(f_x-labels.loc[index])
             else:
                error = error + (f_x-labels.loc[index]) * (f_x-labels.loc[index])
        else:
            # f_y = output(NN_model,pd.DataFrame(counter_example))[0][0]
            if not isClassification:
                # f_y = vio
                f_y = output(NN_model,pd.DataFrame(counter_example))[0][0]
                error = error + (f_y-labels.loc[index]) * (f_y-labels.loc[index])
                counter_y = counter_example[monotonic_index]
                if not np.isclose(abs(f_x-f_y),vio,atol=1e-3) or getViolation(f_x, f_y, direction) < 0:
                    print("not close enough or negative violation")
                    sys.exit(0)
                sum_of_violation = sum_of_violation + getViolation(f_x, f_y, direction)
                violations = violations + [getViolation(f_x, f_y, direction)]
                no_of_counter = no_of_counter + 1
            else:
                f_y = NN_model.predict_classes(pd.DataFrame(counter_example).transpose())[0][0]
                f_y_prob = output(NN_model,pd.DataFrame(counter_example))[0][0]
                error = error + abs(f_y-labels.loc[index])
                counter_y = counter_example[monotonic_index]

                sum_of_violation = sum_of_violation + abs(f_x-f_y_prob)
                violations = violations + [abs(f_x-f_y_prob)]
                no_of_counter = no_of_counter + 1
        
        label_count = label_count + 1

    avg = 0.0
    max_vio = 0.0
    env_metrics_dict = {}
    prediction_metrics_dict = {}
    env_metrics_dict['avg'] = sum(all_results_envelope)/len(all_results_envelope)
    env_metrics_dict['total_time'] = sum(all_results_envelope)
    env_metrics_dict['min'] = min(all_results_envelope)
    env_metrics_dict['max'] = max(all_results_envelope)
    env_metrics_dict['median'] = np.percentile(all_results_envelope, 50)
    env_metrics_dict['first_quartile'] = np.percentile(all_results_envelope, 25)
    env_metrics_dict['third_quartile'] = np.percentile(all_results_envelope, 75)
    # avg_envelope_time = sum(all_results_envelope)/len(all_results_envelope)
    # total_envelope_time = sum(all_results_envelope)
    prediction_metrics_dict['avg'] = sum(all_results_prediction)/len(all_results_prediction)
    prediction_metrics_dict['total_time'] = sum(all_results_prediction)
    prediction_metrics_dict['min'] = min(all_results_prediction)
    prediction_metrics_dict['max'] = max(all_results_prediction)
    prediction_metrics_dict['median'] = np.percentile(all_results_prediction, 50)
    prediction_metrics_dict['first_quartile'] = np.percentile(all_results_prediction, 25)
    prediction_metrics_dict['third_quartile'] = np.percentile(all_results_prediction, 75)
    if len(violations) > 0:
        max_vio = max(violations)
    if no_of_counter > 0 :
        avg = sum_of_violation/no_of_counter*1.0
    if not isClassification:
        return error/len(data), no_of_counter, avg, max_vio, env_metrics_dict,prediction_metrics_dict
    else:
        return (len(data)-error)/len(data),no_of_counter,avg,max_vio,env_metrics_dict,prediction_metrics_dict

#Overapproximation/Envelope error:
def getEnvelopeResult_parallel(data, labels, NN_model, monotonic_index, fold, counter_example_generator, output, configurations, direction):
    dataframe = pd.DataFrame(data)
    label_dataframe = pd.DataFrame(labels)
    num_cores = multiprocessing.cpu_count()
    # num_cores = 8
    num_cores = int(configurations['num_cores'])
    no_chunks = math.ceil(len(data)/num_cores)
    data_chunks = np.array_split(data, no_chunks)
    all_results = []
    with Parallel(n_jobs=num_cores) as parallel:
        for chunk in data_chunks:
            results = parallel(delayed(generate_counter_example)(configurations, counter_example_generator, point.copy(), monotonic_index, index, output(NN_model, point)[0][0], fold) for index,point in chunk.iterrows())
            all_results.extend(results)

    if len(all_results) != len(dataframe):
        raise Exception("Length mismatch of parallel processing")

    return getEnvelopeMetrics(all_results, data, labels, output, NN_model, monotonic_index, direction,configurations['is_classification'])


#per each data point x, find y where x<y and f(x) <f(y) is maximized.
#here we can either use any SMT or MIP solvers to find the counter example
#we use optimathsat, since in our experiments this solver seems to be the fastest
#we could also try both version of fixing the counter example and assume that y is our
#datapoint and we look to find x where x<y and f(x) <f(y) is maximized.
def generate_counter_example(configurations, counter_example_generator, data_point, monotonic_index, data_index,label,fold):
    weights_directory = configurations['weight_files']
    layers = configurations['layers']
    monotonicity_direction =  int(configurations['monotonicity_directions'])
    logging.debug('The monotonic direction is %s'%(monotonicity_direction))
    min_dict = {}
    max_dict = {}
    for monotonic_index in configurations['monotonic_indices']:
        min_max_list = configurations['min_max_values'][configurations['column_names'][monotonic_index]]
        min_dict[monotonic_index] = min_max_list[0]
        max_dict[monotonic_index] = min_max_list[1]
    counter_example, elapsed_time, vio = counter_example_generator(data_point, configurations['monotonic_indices'],label,weights_directory,layers,monotonicity_direction,min_dict,max_dict,configurations["column_types"],logging, configurations['tmp_prefix'], configurations['is_classification'])
    return counter_example, elapsed_time, vio, data_index

def getViolation(f_x, f_y, direction):
  if direction == "upper":
        return float((f_y-f_x))
  else:
        return float((f_x-f_y))