import os
import re
import sys
import fileinput
import importlib
import subprocess

from os import path
from Analysis import analysis
from Utils import readConfigurations


def run(configFile):
    print("Reading from configurations at "+configFile)
    configs = readConfigurations(configFile)
    sys.path.append('./src/Models')

    print("Generating results for "+ str(configs['monotonic_indices']))
    cmd = "python3 src/Verification_framework.py " + configFile
    print("The command to run is %s"%(cmd))
    p = subprocess.Popen(cmd,stderr = subprocess.PIPE,shell=True)
    (output, err) = p.communicate()
    p_status = p.wait()
    print("Finished verifying")
    if configs['is_classification']:
        task_type =  {configs['name'] : "classification"}
    else:
        task_type =  {configs['name'] : "regression"}
    
    analysis(configs['run_data_path'] + configs['name'] + "/", [configs['name']], "train", task_type)
    print("You can find the best counterexample trained model here : " + configs['run_data_path'] + configs['name'])

args = sys.argv
run(args[1])