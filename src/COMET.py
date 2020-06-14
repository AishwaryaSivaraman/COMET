import re
import os
import sys
import argparse
import importlib
import fileinput
import subprocess

from os import path
from Utils import readConfigurations, removeFiles, makeDir, checkifFileExists


parser = argparse.ArgumentParser(description='Counter-example guided Monotonicity Verification Framework')
parser.add_argument('config_file', metavar='c', type=str,
                    help='configuration file')
parser.add_argument('--mode', metavar='m', type=str,
                    help='three modes of operation, \n (1) "verifier" mode checks if a given network is monotonic \n (2) "learner" mode retrains the model with monotonic examples \n (3) "envelope" mode returns monotonic predictions for each test point')
parser.add_argument('--test_file', metavar='f',help='data points file for predictions',required=False)
parser.add_argument('--model_file', metavar='d',help='model file',required=False)

args = parser.parse_args()
config_file = args.config_file
mode = args.mode

if mode == "learner":
  cmd = "python3 src/run.py " + config_file
  print(cmd)
  p = subprocess.Popen(cmd,stderr = subprocess.PIPE,shell=True)
  (output, err) = p.communicate()
  p_status = p.wait()
  if p.returncode != 0:
    print(err)
  print("completed monotonic learning")

if mode == "verifier":
  model_file = args.model_file
  cmd = "python3 src/MonotonicVerifier.py " + config_file
  p = subprocess.Popen(cmd,stderr = subprocess.PIPE,shell=True)
  (output, err) = p.communicate()
  p_status = p.wait()
  if p.returncode != 0:
    print(err)
  print("completed verification")

if mode == "envelope":
  test_file = args.test_file
  print("Predicting...")
  cmd = "python3 src/VerifiedPredictions.py " + config_file + "  " + test_file
  p = subprocess.Popen(cmd,stderr = subprocess.PIPE,shell=True)
  (output, err) = p.communicate()
  p_status = p.wait()
  if p.returncode != 0:
    print(err)
  print("completed envelope predictions")
