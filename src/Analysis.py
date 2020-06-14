import os
import sys
import operator
import matplotlib.pyplot as plt

from math import sqrt
from shutil import copyfile
from statistics import mean
from beautifultable import BeautifulTable
from AnalysisUtils import extractFeatures, getLatestDir, processLogFile

log_data = {}
feature_data = {}

def compare(dataset_task_type, upper, lower):
  if dataset_task_type == "regression":
    if upper > lower:
      return "lower"
    else:
      return "upper"
  else:
    if upper > lower:
      return "upper"
    else:
      return "lower"

def log_extraction(data_dir, benchmarks, reporting_criteria, dataset_task_type):
  global log_data
  global feature_data
  n_folds = 1
  for benchmark in benchmarks:
    feature_data[benchmark] = extractFeatures(data_dir, benchmark)
  
  for benchmark in benchmarks:
    log_data[benchmark] = {}
    for feature in feature_data[benchmark]:
      log_data[benchmark][feature] = {}
      for fold in range(0, n_folds):
        dir = data_dir 
        latest_dir = getLatestDir(dir, feature)
        log_file = latest_dir + "/logs/log.txt"
        processed_data, test_metric, test_cg_metric, train_upper_metric, train_lower_metric = processLogFile(log_file)
        reporting_index = -1
        if reporting_criteria == "envelope":
          if dataset_task_type[benchmark] == "regression":
            reporting_index_upper, value_upper = min(enumerate(train_upper_metric), key=operator.itemgetter(1))
            reporting_index_lower, value_lower = min(enumerate(train_lower_metric), key=operator.itemgetter(1))
            if value_lower < value_upper:
              reporting_index = reporting_index_lower
            else:
              reporting_index = reporting_index_upper
          else:
            reporting_index_upper, value_upper = max(enumerate(train_upper_metric), key=operator.itemgetter(1))
            reporting_index_lower, value_lower = max(enumerate(train_lower_metric), key=operator.itemgetter(1))
            if value_lower < value_upper:
              reporting_index = reporting_index_upper
            else:
              reporting_index = reporting_index_lower
        if reporting_criteria == "train":
          if dataset_task_type[benchmark] == "regression":
            reporting_index, value = min(enumerate(test_metric), key=operator.itemgetter(1))
          else:
            reporting_index, value = max(enumerate(test_metric), key=operator.itemgetter(1))
        if reporting_criteria == "cg":
          reporting_index, value = min(enumerate(test_cg_metric), key=operator.itemgetter(1))
          reporting_index = int(reporting_index/2)
        log_data[benchmark][feature][fold] = []
        log_data[benchmark][feature][fold].append(processed_data[0])
        log_data[benchmark][feature][fold].append(processed_data[reporting_index])
        copyfile(latest_dir + "/folds/model_" + str(reporting_index)+".h5" ,data_dir+"best_model.h5")
        print("Best Model found at Epoch "+ str(reporting_index))
        

def printTable(benchmarks, dataset_task_type):
  n_folds = 1
  global log_data
  global feature_data
  baseline_envelope = {}
  table = BeautifulTable()
  table.column_headers = [" ", "Original", "CGL", "Original Envelope", "COMET Envelope"]
  for benchmark in benchmarks:
    baseline_envelope[benchmark] = {}
    for feature in feature_data[benchmark]:
      total_test = []
      total_train = []
      total_envelope_test = []
      total_envelope_train = []
      baseline_envelope[benchmark][feature] = []
      for fold in range(0, n_folds):
        comp = compare(dataset_task_type[benchmark], log_data[benchmark][feature][fold][0].test.upper.envelope_quality, log_data[benchmark][feature][fold][0].test.lower.envelope_quality)
        if comp == "upper":
          total_test.append(log_data[benchmark][feature][fold][0].test.upper.baseline_quality)
          total_envelope_test.append(log_data[benchmark][feature][fold][0].test.upper.envelope_quality)
          total_train.append(log_data[benchmark][feature][fold][0].train.upper.baseline_quality)
          total_envelope_train.append(log_data[benchmark][feature][fold][0].train.upper.envelope_quality)
        else:
          total_test.append(log_data[benchmark][feature][fold][0].test.lower.baseline_quality)
          total_envelope_test.append(log_data[benchmark][feature][fold][0].test.lower.envelope_quality)
          total_train.append(log_data[benchmark][feature][fold][0].train.lower.baseline_quality)
          total_envelope_train.append(log_data[benchmark][feature][fold][0].train.lower.envelope_quality)
        avg_test = str(round(mean(total_test),2))
        avg_env_test = str(round(mean(total_envelope_test),2))
        avg_train = str(round(mean(total_train),2))
        avg_env_train = str(round(mean(total_envelope_train),2))
        comp = compare(dataset_task_type[benchmark], log_data[benchmark][feature][fold][1].test.upper.envelope_quality, log_data[benchmark][feature][fold][1].test.lower.envelope_quality)
        
        total_test = []
        total_train = []
        total_envelope_test = []
        total_envelope_train = []

        total_test.append(log_data[benchmark][feature][fold][1].test.upper.baseline_quality)
        total_train.append(log_data[benchmark][feature][fold][1].train.upper.baseline_quality)
        
        if comp == "upper":
          print("For best performance and monotonic predictions, use best_model.h5 with upper envelope")
          total_envelope_test.append(log_data[benchmark][feature][fold][1].test.upper.envelope_quality)
          total_envelope_train.append(log_data[benchmark][feature][fold][1].train.upper.envelope_quality)
        else:
          print("For best performance and monotonic predictions, use best_model.h5 with lower envelope")
          total_envelope_test.append(log_data[benchmark][feature][fold][1].test.lower.envelope_quality)
          total_envelope_train.append(log_data[benchmark][feature][fold][1].train.lower.envelope_quality)
        avg_comet_test = str(round(mean(total_test),2))
        avg_comet_train = str(round(mean(total_train),2))
        avg_comet_env_test = str(round(mean(total_envelope_test),2))
        avg_comet_env_train = str(round(mean(total_envelope_train),2))

      table.append_row(["Train", avg_train, avg_comet_train, avg_env_train, avg_comet_env_train])
      table.append_row(["Test", avg_test, avg_comet_test, avg_env_test, avg_comet_env_test])
  print(table)


def baselineVsComet(n_folds, benchmarks, dataset_task_type, output_type):
  global log_data
  global feature_data
  baseline_comet = {}
  for benchmark in benchmarks:
    baseline_comet[benchmark] = {}
    for feature in feature_data[benchmark]:
      total_test = []
      total_train = []
      total_envelope_test = []
      total_envelope_train = []
      baseline_comet[benchmark][feature] = []
      for fold in range(0, n_folds):
        comp = compare(dataset_task_type[benchmark], log_data[benchmark][feature][fold][1].test.upper.envelope_quality, log_data[benchmark][feature][fold][1].test.lower.envelope_quality)
        if comp == "upper":
          total_test.append(log_data[benchmark][feature][fold][0].test.upper.baseline_quality)
          total_envelope_test.append(log_data[benchmark][feature][fold][1].test.upper.baseline_quality)
          total_train.append(log_data[benchmark][feature][fold][0].train.upper.baseline_quality)
          total_envelope_train.append(log_data[benchmark][feature][fold][1].train.upper.baseline_quality)
        else:
          total_test.append(log_data[benchmark][feature][fold][0].test.lower.baseline_quality)
          total_envelope_test.append(log_data[benchmark][feature][fold][1].test.lower.baseline_quality)
          total_train.append(log_data[benchmark][feature][fold][0].train.lower.baseline_quality)
          total_envelope_train.append(log_data[benchmark][feature][fold][1].train.lower.baseline_quality)
      if len(total_test) > 1:
        avg_test = str(round(mean(total_test),2)) + r"$\pm$" + str(round(stdev(total_test),2))
        avg_env_test = str(round(mean(total_envelope_test),2)) + r"$\pm$" + str(round(stdev(total_envelope_test),2))
        avg_train = str(round(mean(total_train),2)) + r"$\pm$" + str(round(stdev(total_train),2))
        avg_env_train = str(round(mean(total_envelope_train),2)) + r"$\pm$" + str(round(stdev(total_envelope_train),2))
      else:
        avg_test = str(round(mean(total_test),2))
        avg_env_test = str(round(mean(total_envelope_test),2))
        avg_train = str(round(mean(total_train),2))
        avg_env_train = str(round(mean(total_envelope_train),2))
      print(total_envelope_test)
      baseline_comet[benchmark][feature].append(avg_train)
      baseline_comet[benchmark][feature].append(avg_env_train)
      baseline_comet[benchmark][feature].append(avg_test)
      baseline_comet[benchmark][feature].append(avg_env_test)
  if output_type == "Text":
    print("Baseline vs COMET")
    print(printText(baseline_comet))

def analysis(data_dir, benchmarks, reporting_criteria, dataset_task_type):
  log_extraction(data_dir, benchmarks, reporting_criteria, dataset_task_type)
  printTable(benchmarks,dataset_task_type)



