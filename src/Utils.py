import os
import csv
import shutil
import shutil, errno

from ast import literal_eval

def deleteFolder(src):
  shutil.rmtree(src)

def copyfiles(src, dst):
    try:
       if os.path.exists(dst):
        shutil.rmtree(dst)
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise

def makeDir(path, recursive = False):
    if not os.path.exists(path):
        if recursive:
          os.makedirs(path,exist_ok=True)
        else:
            os.mkdir(path)
  
def write_to_csv(filename, content):
  with open(filename, mode='a') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=' ')
    csv_writer.writerow(content)


def readModelConfigurations (config_file):
    f = open(config_file, "r")
    configuration = {}
    for x in f:
        key = x.split(" = ")[0]
        value_string = x.split(" = ")[1].rstrip()
        value = literal_eval(value_string)
        configuration[key] = value
    return configuration

def readConfigurations (config_file):
    f = open(config_file, "r")
    configuration = {}
    configuration["n_folds"] = 1
    configuration["counter_example_type"] = "cg"
    configuration["solve_separate"] = 0
    for x in f:
        key = x.split(" = ")[0]
        value = x.split(" = ")[1].rstrip()
        if key == "layers" or key == "monotonicity_direction" or key == "re-run" or key =="n_folds" or key == "solve_separate" or key == "number_of_epochs":
            value = int(value)
        if key == "column_names" or key=="column_types" or key == "feature_names" or key == "feature_types":
            if "," in value:
                value = value.split(',')
            else:
                value = [value]
        if key == "monotonic_indices" or key == "indices" or key == "mon_dir":
            if "," in value:
                value = list(map(int, value.split(',')))
            else:
                value = [int(value)]
        if key == "model_parameters":
            value = literal_eval(value)
        if key == "min_max_values":
            value = literal_eval(value)
        if value == "True" or value == "False":
            value = eval(value)
        configuration[key] = value
    configuration['layers'] = configuration['model_parameters']['layer_size']
    configuration['current_benchmark'] = configuration['name']
    configuration['fold_data_dir'] = configuration['model_dir']
    configuration['column_names'] = configuration['feature_names']
    configuration['column_types'] = configuration['feature_types']
    configuration['retrain_model'] = True
    return configuration

def copyFile(fromFolder, toFolder):
    src_files = os.listdir(fromFolder)
    for file_name in src_files:
        full_file_name = os.path.join(fromFolder, file_name)
        full_old_name = os.path.join(toFolder, file_name)
        if os.path.isfile(full_old_name):
            os.remove(full_old_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, toFolder)

def checkifFileExists(folder, name):
    if os.path.isdir(folder):
        src_files = os.listdir(folder)
        for file_name in src_files:
            if name in file_name:
                return True
    return False
        
def removeFiles(folder):
    if os.path.isdir(folder):
        src_files = os.listdir(folder)
        for file_name in src_files:
            full_file_name = os.path.join(folder, file_name)
            if os.path.isfile(full_file_name):
                os.remove(full_file_name)