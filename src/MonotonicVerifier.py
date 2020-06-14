#!/usr/bin/env python
# coding: utf-8


import os
import sys
import time
import random
import argparse
import importlib
import numpy as np

from Solver_OptiMathSat import verifier
from Envelope import getEnvelopeResult, generate_counter_example
from Utils import makeDir,write_to_csv,copyfiles,copyFile, readConfigurations
from ModelCalls import generate_data, make_batch, evaluate_model, update_model


def monotonicity_verifier():
    configurations = getConfigurations()
    print("Verifying dataset : " + configurations['name'])
    print("#Layers: " + str(configurations['model_parameters']['layer_size']) + " #Neurons/layer: " + str(configurations['model_parameters']['hidden_size']))
    initialModel = configurations['fold_data_dir']
    weights_directory = os.path.dirname(model_file)+'/'
    setConfigurations('weight_files',weights_directory)
    column_names: List[str] = configurations['column_names']
    layers = configurations['layers']
    monotonicity_direction =  int(configurations['monotonicity_directions'])
    min_dict = {}
    max_dict = {}
    for index in range(0, len(configurations['column_names'])):
        min_max_list = configurations['min_max_values'][configurations['column_names'][index]]
        min_dict[index] = min_max_list[0]
        max_dict[index] = min_max_list[1]
    result, returned_value, elapsed_time = verifier(configurations['monotonic_indices'],weights_directory,layers,monotonicity_direction,min_dict,max_dict,configurations["column_types"],"", configurations['tmp_prefix'])

    if "UNSAT" in result:        
        print("Network for %s is monotonic"%(configurations['name'])) 
        print("Time taken (s) " + str(elapsed_time))
    else:
        print("Network for %s is not monotonic"%(configurations['name'])) 
        print("Time taken (s) " + str(elapsed_time))

def setConfigurations (key,value):
    global configurations
    configurations[key] = value

def getConfigurations ():
    global configurations
    return configurations

parser = argparse.ArgumentParser(description='Monotonicity Verifier')
parser.add_argument('config_file', metavar='c', type=str,
                    help='configuration file')

args = parser.parse_args()
config_file = args.config_file
configurations = readConfigurations(config_file)
model_file = configurations['model_dir']

#-------- Verification ---------
monotonicity_verifier()
