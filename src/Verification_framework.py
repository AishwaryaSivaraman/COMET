#!/usr/bin/env python
# coding: utf-8

import sys
import math
import time
import random
import pickle
import logging
import argparse
import importlib
import statistics
import numpy as np
import pandas as pd
import multiprocessing
import tensorflow as tf
import matplotlib.pyplot as plt

from typing import List
from ast import literal_eval
from datetime import datetime
from joblib import Parallel, delayed
from Envelope import getEnvelopeResult, generate_counter_example
from ModelCalls import generate_data, make_batch, evaluate_model, update_model
from Utils import makeDir, write_to_csv, copyfiles, copyFile, readConfigurations



def writeEnvelopeResutlsToFile(print_string, error, no_cg, data_size, avg_violation, max_violation, log, print_count):
    print('%d,%s envelope, %.4f'%(print_count, print_string, error), file=log)
    print('%d,%s cgs/total points, %d ,%d'%(print_count, print_string, no_cg, data_size), file=log)
    print('%d,%s average violation, %.4f'%(print_count, print_string, avg_violation), file=log)
    print('%d,%s maximum violation, %.4f'%(print_count, print_string, max_violation), file=log)

def collectEnvelopeMetric(NN_model, data, labels, log, monotonic_index, fold, direction, print_string, counter_example_generator, print_count):
    print('%d,%s, %.4f'%(print_count, print_string, evaluate_model(NN_model, data, labels, getConfigurations(), evaluate)), file=log)
    if getConfigurations()['scalability']:
        error, no_cg, avg_violation, max_violation, env_metrics_dict,prediction_metrics_dict = getEnvelopeResult(data, labels, NN_model, monotonic_index,fold, getConfigurations()["is_parallel"], counter_example_generator, output, getConfigurations(), direction)
        print('%d,%s envelope metrics, %s'%(print_count, print_string+" "+direction, str(env_metrics_dict)), file=log)
        print('%d,%s prediction metrics, %s'%(print_count, print_string+" "+direction,str(prediction_metrics_dict)), file=log)
    else:
        error, no_cg, avg_violation, max_violation = getEnvelopeResult(data, labels, NN_model, monotonic_index,fold, getConfigurations()["is_parallel"], counter_example_generator, output, getConfigurations(), direction)
    writeEnvelopeResutlsToFile(print_string+" "+direction, error, no_cg, len(data), avg_violation, max_violation, log, print_count)
    return no_cg

def get_counter_example_u_l(data_point, monotonic_index, data_index, f_x, fold):
    counter_example_upper,elapsed_time_u,vio_u,ind_u = generate_counter_example(getConfigurations(), counter_example_generator_upper, data_point.copy(), monotonic_index, data_index, f_x, fold)

    counter_example_lower,elapsed_time_l,vio_l,ind_l = generate_counter_example(getConfigurations(), counter_example_generator_lower, data_point.copy(), monotonic_index, data_index, f_x, fold)

    return counter_example_upper, vio_u, ind_u, counter_example_lower, vio_l,ind_l

def verification(data_path, log_file, n_folds, monotonic_indices):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=getConfigurations()['run_data_path']+"app.log", filemode='w', format='%(name)s - %(levelname)s - %(message)s',level = logging.DEBUG)
    logger = logging.getLogger('nnverification')

    global solver_times
    logging.info("Generating Data for train and test")
    
    train_all, train_labels_all, test_all, test_labels_all,min_max_dict = generate_data(make_data, getConfigurations())

    setConfigurations('min_max_values',min_max_dict)

    column_names: List[str] = getConfigurations()['column_names']

    fold = 0
    initialModel = getConfigurations()['fold_data_dir']
    copyFile(initialModel, getConfigurations()['weight_files'])

    monotonic_index = monotonic_indices
    start_time = time.time()
    train_data = train_all[fold]
    train_labels = train_labels_all[fold]
    test_data = test_all[fold]
    test_labels = test_labels_all[fold]
    logging.debug("Mean Squared Error after initial training: \n")
    model_file = configurations['model_dir'] + 'model.h5'
    NN_model = tf.keras.models.load_model(model_file)

    logging.debug("The train error is ")
    logging.debug(evaluate_model(NN_model,train_data,train_labels,getConfigurations(), evaluate))
    print("The train error is ")
    print(evaluate_model(NN_model,train_data,train_labels,getConfigurations(), evaluate))
    logging.debug("The test error is ")
    logging.debug(evaluate_model(NN_model,test_data,test_labels,getConfigurations(),evaluate))
    print("The test error is")
    print(evaluate_model(NN_model,test_data,test_labels,getConfigurations(),evaluate))

    batches = make_batch(train_data, train_labels, getConfigurations())
    batch_index = 0
    temp_batch_count =0
    plot_test_data =[]
    plot_test_label=0
    for index in batches[batch_index]:
        plot_test_data = train_data.iloc[index].values
        plot_test_label = train_labels.iloc[index]

    with open(log_file+'log.txt',"a") as log:
        # Stats with initial model:
        NN_model.save(getConfigurations()['weight_files']+"model_0.h5")
        isRetrainWCG = True
        print_count = 0
        start_env_time = time.time()
        no_cg_upper = collectEnvelopeMetric(NN_model, test_data, test_labels, log, monotonic_index, fold, "upper", "test",counter_example_generator_upper,print_count)
        elapse_env_time = time.time() - start_env_time
        writeTimeToFile(time.strftime("%H:%M:%S", time.gmtime(elapse_env_time)),fold,"Test Upper Envelope time : ")

        start_env_time = time.time()
        no_cg_lower = collectEnvelopeMetric(NN_model, test_data, test_labels, log, monotonic_index, fold, "lower", "test",counter_example_generator_lower,print_count)
        elapse_env_time = time.time() - start_env_time
        writeTimeToFile(time.strftime("%H:%M:%S", time.gmtime(elapse_env_time)),fold,"Test Lower Envelope time : ")

        isNoViolationTest = False
        no_cg = no_cg_upper + no_cg_lower
        if no_cg == 0:
            isNoViolationTest = True

        start_env_time = time.time()
        no_cg_upper = collectEnvelopeMetric(NN_model, train_data, train_labels, log, monotonic_index, fold, "upper", "train", counter_example_generator_upper,print_count)
        elapse_env_time = time.time() - start_env_time
        writeTimeToFile(time.strftime("%H:%M:%S", time.gmtime(elapse_env_time)),fold,"Train Upper Envelope time : ")

        start_env_time = time.time()
        no_cg_lower = collectEnvelopeMetric(NN_model, train_data, train_labels, log, monotonic_index, fold, "lower", "train", counter_example_generator_lower,print_count)
        elapse_env_time = time.time() - start_env_time
        writeTimeToFile(time.strftime("%H:%M:%S", time.gmtime(elapse_env_time)),fold,"Train Lower Envelope time : ")

        no_cg = no_cg_upper + no_cg_lower

        isNoViolationTrain = False

        if no_cg == 0:
            isNoViolationTrain = True
            isRetrainWCG = False

        setConfigurations("scalability", False)
        if getConfigurations()['counter_example_type'] == "cg" and getConfigurations()['retrain_model']:
            logging.debug('Counter Example Learning')
            print('Counter Example Learning')
            for epoch in range(0,getConfigurations()['number_of_epochs']):
                if not isRetrainWCG:
                    logging.debug('No violations in train and test!')
                    print('No violations in train and test!')
                    NN_model = update_model(train_data, train_labels, NN_model,getConfigurations(),update_batch)
                    isRetrainWCG = True
                    # break
                logging.debug('Starting epoch: %d'%(epoch))
                print('Starting epoch: %d'%(epoch))

                temp_batch_count = 0
                number_counter_unsat = 0
                
                for batch in batches:
                    counter_example_count = 0
                    temp_batch_count = temp_batch_count+1

                    logging.debug('Progress... batch/batches:%d/%d'%(temp_batch_count,len(batches)))
                    counter_batch = []
                    batch_labels = []
                    count = 0
                    num_cores = int(getConfigurations()['num_cores'])
                    all_results = []
                    with Parallel(n_jobs=num_cores) as parallel:
                        results = parallel(delayed(get_counter_example_u_l)(train_data.iloc[data_index].values, monotonic_index, data_index, output(NN_model, train_data.iloc[data_index].values)[0][0], fold) for data_index in batch)
                        all_results.extend(results)

                    for res in results:
                        try:
                            count = count+1
                            counter_example_upper, vio_u, ind_u, counter_example_lower, vio_l,ind_l = res
                            if ind_u != ind_l:
                                print("The indices dont match")
                                sys.exit(0)
                            data_point = train_data.iloc[ind_u].values
                            logging.debug('CounterExample Progress... count/batchsize:%d/%d'%(count,len(batch)))

                            counter_examples = []

                            counter_examples.append(counter_example_upper)
                            counter_examples.append(counter_example_lower)

                            if getConfigurations()['is_classification']:
                                f_x = NN_model.predict_classes(pd.DataFrame(data_point).transpose())[0][0]
                                avg_f_x = f_x
                            else:
                                f_x = output(NN_model, data_point)[0][0]
                                f_x_cgs = []
                                f_x_cgs.append(f_x)
                                for counter_example in counter_examples:
                                    if counter_example is not None:
                                        f_x_cgs.append(output(NN_model, counter_example)[0][0])
                                avg_f_x = 1.0*sum(f_x_cgs)/len(f_x_cgs)

                            counter_batch.append(data_point)
                            batch_labels.append(avg_f_x)

                            for counter_example in counter_examples:
                                if counter_example is None:
                                    number_counter_unsat = number_counter_unsat + 1
                                else:
                                    counter_example_count = counter_example_count+1
                                    counter_batch.append(counter_example)
                                    batch_labels.append(avg_f_x)
                        except:
                            print("Exception while processing counterexample " + sys.exc_info()[0])
                            sys.exit(0)
                    original_train = train_data
                    original_label = train_labels

                    if len(counter_batch) != len(batch_labels):
                        print("Length of cg and label not equal")
                        sys.exit(0)

                    if (len(counter_batch)>0):
                        counter_batch = pd.DataFrame(counter_batch, columns= column_names)
                        batch_labels = pd.DataFrame(batch_labels)

                        train_batch = original_train.append(counter_batch, ignore_index = True,sort=False)
                        train_batch_label = original_label.append(batch_labels, ignore_index = True)

                        logging.debug('Mean Squared Error after batch %d/%d counterexample: '%((temp_batch_count,len(batches))))
                        NN_model = update_model(train_batch, train_batch_label, NN_model,getConfigurations(),update_batch)


                    batch_index+=1

                logging.debug("The model after epoch "+str(epoch))
                logging.debug(evaluate_model(NN_model, train_data,train_labels,getConfigurations(),evaluate))
                # Save the model after each epoch
                NN_model.save(getConfigurations()['weight_files']+"model_"+str(epoch+1)+".h5")
                # Logging metrics after each epoch:
                print_count = epoch+1
                no_cg_upper = collectEnvelopeMetric(NN_model, test_data, test_labels, log, monotonic_index, fold, "upper", "test",counter_example_generator_upper, print_count)

                no_cg_lower = collectEnvelopeMetric(NN_model, test_data, test_labels, log, monotonic_index, fold, "lower", "test",counter_example_generator_lower, print_count)


                no_cg_upper = collectEnvelopeMetric(NN_model, train_data, train_labels, log, monotonic_index, fold, "upper", "train", counter_example_generator_upper, print_count)

                no_cg_lower = collectEnvelopeMetric(NN_model, train_data, train_labels, log, monotonic_index, fold, "lower", "train", counter_example_generator_lower, print_count)

                no_cg = no_cg_upper + no_cg_lower

                if no_cg == 0:
                    isRetrainWCG = False

        if not isRetrainWCG and getConfigurations()['retrain_model']:
            #Jump here if both train and test errors are 0
            print('No violation in test and train', file=log)
            #plot graphs for a set of random points and check if it is indeed monotonic:
            #Generate random points from test and train

        if getConfigurations()['counter_example_type'] == "cg_pair" and getConfigurations()['retrain_model']:
            logging.debug('CounterPair Learning')
            print('CounterPair Learning')
            pair_count = 0
            epoch_count = int(getConfigurations()['number_of_epochs'])
            while (pair_count < epoch_count):
                print("Starting epoch "+str(pair_count))
                counter_batch, batch_labels,violations = counter_example_pairs(NN_model, monotonic_index, column_names, train_data, train_labels,logging)
                if len(counter_batch) == 0:
                    print('No counter pair violation in train', file=log)
                    NN_model = update_model(train_data, train_labels, NN_model, getConfigurations(), update_batch)
                    # break
                else:
                    counter_batch = pd.DataFrame(counter_batch, columns= column_names)
                    batch_labels = pd.DataFrame(batch_labels)
                    original_train = train_data
                    original_label = train_labels
                    train_batch = original_train.append(counter_batch, ignore_index = True) 
                    train_batch_label = original_label.append(batch_labels, ignore_index = True)
                    print("Size of counter batch "+str(len(counter_batch)))
                    NN_model = update_model(train_batch, train_batch_label, NN_model, getConfigurations(), update_batch)
                    # Logging metrics after each counter_pair learning:
                    avg = (sum(violations) * 1.0)/len(violations)
                    print('%s,cp, number of counter pairs / total count, %d %d'%('cp'+str(pair_count), len(violations), len(train_data)), file=log)
                    print('%s,cp, average violation of counter pairs, %.4f'%('cp'+str(pair_count), avg), file=log)
                    print('%s,cp, maximum violation of counter pairs, %.4f'%('cp'+str(pair_count), max(violations)), file=log)

                    # Logging metrics after each epoch:
                    print_count = pair_count+1
                    no_cg_upper = collectEnvelopeMetric(NN_model, test_data, test_labels, log, monotonic_index, fold, "upper", "test",counter_example_generator_upper, print_count)

                    no_cg_lower = collectEnvelopeMetric(NN_model, test_data, test_labels, log, monotonic_index, fold, "lower", "test",counter_example_generator_lower, print_count)


                    no_cg_upper = collectEnvelopeMetric(NN_model, train_data, train_labels, log, monotonic_index, fold, "upper", "train", counter_example_generator_upper, print_count)

                    no_cg_lower = collectEnvelopeMetric(NN_model, train_data, train_labels, log, monotonic_index, fold, "lower", "train", counter_example_generator_lower, print_count)

                pair_count = pair_count + 1

        elapsed_time = time.time() - start_time
        print("total elapsetime is: ")
        print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        writeTimeToFile(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)),fold)
        average_time = 0.0
        if len(solver_times) > 0:
            average_time =cl (sum(solver_times) / len(solver_times))
        writeTimeToFile(time.strftime("%H:%M:%S", time.gmtime(average_time)),fold,"Average Time taken to solve each query by solver "+getConfigurations()['solver_name'])

def get_monoticity_direction(monotonic_index):
    index = getConfigurations()['monotonic_indices'].index(str(monotonic_index))
    return getConfigurations()['monotonicity_directions'][index]


def counter_example_pairs(NN_model, monotonic_index, column_names, train_data, train_labels, logging):
    if getConfigurations()['is_parallel']:
        try:
            return counter_example_pairs_parallel(NN_model, monotonic_index, column_names, train_data, train_labels, logging)
        except Exception as e:
            print(e)
            print("Parallel processing for counter pairs failed, trying sequential")
            return counter_example_pairs_seq(NN_model, monotonic_index, column_names, train_data, train_labels, logging)
    else:
        return counter_example_pairs_seq(NN_model, monotonic_index, column_names, train_data, train_labels, logging)


def counter_example_pairs_parallel(NN_model, monotonic_index, column_names, train_data, train_labels, logging):
    counter_batch = []
    batch_labels = []
    violation = []
    variable_size = train_data.shape[1]
    weights_directory = getConfigurations()['weight_files']
    layers = getConfigurations()['layers']
    monotonicity_direction = get_monoticity_direction(monotonic_index)
    min_max_list = getConfigurations()['min_max_values'][getConfigurations()['column_names'][monotonic_index]]
    print_count = 0

    num_cores = multiprocessing.cpu_count()
    num_cores = int(getConfigurations()['num_cores'])
    print(num_cores)
    print(len(train_data))
    no_chunks = math.ceil(len(train_data)/num_cores)
    print(no_chunks)
    _data_chunks = np.array_split(train_data, no_chunks)
    all_results = []
    with Parallel(n_jobs=num_cores) as parallel:
        for chunk in _data_chunks:
            results = parallel(delayed(counter_pair_generator)(point.copy(), monotonic_index, index, 0,weights_directory, layers, monotonicity_direction,min_max_list[0],min_max_list[1],getConfigurations()["column_types"],"",logging,True,getConfigurations()['tmp_prefix']) for index,point in chunk.iterrows())
            all_results.extend(results)
    
    if len(all_results) != len(train_data):
        print("Length mismatch")
        raise Exception("Length mismatch of parallel processing")

    for res in all_results:
        counter_pair,elapsed_time, index = res
        point = train_data.loc[index].values

        if not counter_pair == None:
            point_x = point.copy()
            point_y = point.copy()
            point_x[monotonic_index] = counter_pair[0]
            point_y[monotonic_index] = counter_pair[1]
            output_x = output(NN_model, point_x)[0][0]
            output_y = output(NN_model, point_y)[0][0]
            avg_f_x = (output_x + output_y)/2
            violation.append(abs(1.0*(output_x-output_y)))
            #     #We swap the labels
            # print("X is "+str(counter_pair[0]))
            # print("X' is "+str(counter_pair[1]))
            # print("Y is "+str(output_y))
            # print("Y' is "+str(output_x))

            # counter_batch.append(point_x)
            # counter_batch.append(point_y)
            # batch_labels.append(output_y)
            # batch_labels.append(output_x)
            counter_batch.append(point_x)
            counter_batch.append(point_y)
            batch_labels.append(avg_f_x)
            batch_labels.append(avg_f_x)
    print(violation)
    return counter_batch, batch_labels,violation
    

def counter_example_pairs_seq(NN_model, monotonic_index, column_names, train_data, train_labels, logging):
    counter_batch = []
    batch_labels = []
    violation = []
    variable_size = train_data.shape[1]
    weights_directory = getConfigurations()['weight_files']
    layers = getConfigurations()['layers']
    monotonicity_direction = get_monoticity_direction(monotonic_index)
    min_max_list = getConfigurations()['min_max_values'][getConfigurations()['column_names'][monotonic_index]]
    print_count = 0
    for index, point in train_data.iterrows():
        print_count = print_count + 1
        counter_pair,elapsed_time,ind = counter_pair_generator(point, monotonic_index, index, 0,weights_directory, layers, monotonicity_direction,min_max_list[0],min_max_list[1],getConfigurations()["column_types"],"",logging,True,getConfigurations()['tmp_prefix'])
        if not counter_pair == None:
            point_x = point.copy()
            point_y = point.copy()
            point_x[monotonic_index] = counter_pair[0]
            point_y[monotonic_index] = counter_pair[1]
            output_x = output(NN_model, point_x)[0][0]
            output_y = output(NN_model, point_y)[0][0]
            violation.append(abs(1.0*(output_x-output_y)))
            #     #We swap the labels
            # print("X is "+str(counter_pair[0]))
            # print("X' is "+str(counter_pair[1]))
            # print("Y is "+str(output_y))
            # print("Y' is "+str(output_x))

            counter_batch.append(point_x)
            counter_batch.append(point_y)
            batch_labels.append(output_y)
            batch_labels.append(output_x)

    return counter_batch, batch_labels,violation
    # counter_pair = counter_pair_generator(variable_size, monotonic_index, MIP_model, data_index,fold,weights_directory,layers,monotonicity_direction)
    
    # if counter_pair[0] == None:
    # return None, None
    # else: 
    #     counter_batch.append(counter_pair[0])
    #     counter_batch.append(counter_pair[1])
    #     counter_score_0 = output(NN_model, counter_pair[0])[0]
    #     counter_score_1 = output(NN_model, counter_pair[1])[0]
    #     #We swap the labels
    #     batch_labels.append(counter_score_1)
    #     batch_labels.append(counter_score_0)
    #     counter_batch = pd.DataFrame(counter_batch, columns= column_names)
    #     batch_labels = pd.DataFrame(batch_labels)
    #     original_train = train_data
    #     original_label = train_labels
    #     test_batch = original_train.append(counter_batch, ignore_index = True) 
    #     test_batch_label = original_label.append(batch_labels, ignore_index = True)
    #     print('Mean Squared Error after pair training')
    #     NN_model,  MIP_model = update_model(test_batch, test_batch_label, NN_model, MIP_model,fold)
    #     return NN_model,  MIP_model   


def writeTimeToFile(time,fold,text=""):
    directory = getConfigurations()['log_files']
    f = open(directory+"timetaken.txt", "a")
    if not text or text=="":
        f.write("Time taken for fold "+str(fold)+" is : "+str(time) +"\n")
    else:
        f.write(text+" is : "+str(time) +"\n")
    f.close()

def setupfolders():
    run_data_dir = getConfigurations()['run_data_path']
    log_files_path = run_data_dir + "logs/"
    makeDir(log_files_path)
    setConfigurations('log_files',log_files_path)

    plot_files_path = run_data_dir + "plots/"
    makeDir(plot_files_path)
    setConfigurations('plot_files',plot_files_path)

    weight_files_path = run_data_dir + "folds/"
    makeDir(weight_files_path)
    setConfigurations('weight_files',weight_files_path)

def setConfigurations (key,value):
    global configurations
    configurations[key] = value

def getConfigurations ():
    global configurations
    return configurations

def monotonicity_verification():
    if int(getConfigurations()['solve_separate']) == 1:
        for m in getConfigurations()['monotonic_indices']:
            dirname = datetime.now().strftime('%Y%m%d')+ str(datetime.now().hour)
            run_data_dir = getConfigurations()['run_data_path']+ getConfigurations()['current_benchmark'] +"/" + getConfigurations()['column_names'][int(m)]+"/"+dirname +"/"
            makeDir(run_data_dir,True)
            setConfigurations('run_data_path',run_data_dir)
            setupfolders()
            configurations = getConfigurations()
            verification(configurations['model_dir'], configurations['log_files'], configurations['n_folds'], getConfigurations()['monotonic_indices'])
    else:
        dirname = datetime.now().strftime('%Y%m%d')+ str(datetime.now().hour)
        dir_col_string = "Combination"
        for m in getConfigurations()['monotonic_indices']:
            dir_col_string = dir_col_string + "+" + getConfigurations()['column_names'][int(m)]
        run_data_dir = getConfigurations()['run_data_path']+ getConfigurations()['current_benchmark'] +"/" + dir_col_string +"/"+dirname +"/"
        makeDir(run_data_dir,True)
        setConfigurations('run_data_path',run_data_dir)
        setupfolders()
        configurations = getConfigurations()
        verification(configurations['model_dir'], configurations['log_files'], configurations['n_folds'], getConfigurations()['monotonic_indices'])

parser = argparse.ArgumentParser(description='Neural Network Verification Framework')
parser.add_argument('config_file', metavar='c', type=str,
                    help='configuration file')

args = parser.parse_args()
config_file = args.config_file
configurations = readConfigurations(config_file)
solver_times = []

#--- import the nn model --------
sys.path.append('./src/Models')
output = importlib.__import__(getConfigurations()['model']).output
evaluate = importlib.__import__(getConfigurations()['model']).evaluate
make_data = importlib.__import__(getConfigurations()['model']).make_data
update_batch = importlib.__import__(getConfigurations()['model']).update_batch

#----- import solver functions --------
counter_example_generator_upper = importlib.__import__(getConfigurations()['solver']).counter_example_generator_upper_env
counter_example_generator_lower = importlib.__import__(getConfigurations()['solver']).counter_example_generator_lower_env

counter_pair_generator = importlib.__import__(getConfigurations()['solver']).counter_pair_generator

#-------- COMET ---------
monotonicity_verification()
