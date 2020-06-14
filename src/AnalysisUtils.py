import os
import sys

class Data:

  def __init__(self, baseline_quality, envelope_quality,no_cgs, total_points):
    self.baseline_quality = baseline_quality
    self.envelope_quality = envelope_quality
    self.no_cgs = no_cgs
    self.total_points = total_points
  
  def display_record(self):
    print("\t\t\t\t Baseline Quality : " + str(self.baseline_quality))
    print("\t\t\t\t Envelope Quality : " + str(self.envelope_quality))
    print("\t\t\t\t # Counterexample : " + str(self.no_cgs))
    print("\t\t\t\t # Total Points : " + str(self.total_points))

class Both:
  def __init__(self, upper, lower):
    self.upper = upper
    self.lower = lower
  
  def display_record(self):
    print("\t\tThe upper envelope detail is:")
    self.upper.display_record()
    print("\t\tThe lower envelope detail is:")
    self.lower.display_record()

class RunData:
  def __init__ (self, test, train):
    self.test = test
    self.train = train

  def display_record(self):
    print("The test detail is:")
    self.test.display_record()
    print("The train detail is:")
    self.train.display_record()

def extractFeatures(data_dir, benchmark):
  src_files = os.listdir(data_dir)
  if '.DS_Store' in src_files:
    src_files.remove('.DS_Store')
  return src_files

def getLatestDir(dir, folder):
  all_subdirs = [f.path for f in os.scandir(os.path.join(dir,folder)) if f.is_dir() ]
  return max(all_subdirs, key=os.path.getmtime)

def processLogFile(log_file):
  print("Processing " + log_file)
  train_metric =[]
  train_cg_metric = []
  train_upper_metric = []
  train_lower_metric = []
  processed_data = {}
  with open(log_file, "r") as ins:
    content = ins.read()
    containsScalability = False
    if "metrics" in content:
      containsScalability = True
  with open(log_file, "r") as ins:
    irofile = iter(ins)
    for line in irofile:
      if 'No violation' in line:
        continue
      baseline_quality= 0.0
      envelope_quality = 0.0
      no_cgs = 0
      total_points = 0
      line_content = line.split(",")
      index = int(line_content[0])
      baseline_quality = float(line_content[2])
      if "train," in line:
        train_metric.append(baseline_quality)
      loop_count = 0
      if index == 0 and containsScalability:
        loop_count = 13
      else:
        loop_count = 9
      while loop_count > 0:
        full_line = next(irofile)
        inner_line = full_line.split(",")
        loop_count = loop_count - 1
        if "envelope," in full_line:
          envelope_quality = float(inner_line[2])
        if "cgs/total" in full_line:
          no_cgs = int(inner_line[2])
          if "train," in line:
            train_cg_metric.append(no_cgs)
          total_points = int(inner_line[3])
        if "train," in full_line or "test," in full_line:
          upper = Data(baseline_quality, envelope_quality, no_cgs, total_points)
          if "train," in full_line:
            train_upper_metric.append(envelope_quality)
      lower = Data(baseline_quality, envelope_quality,no_cgs, total_points)
      if "train," in line:
            train_lower_metric.append(lower.envelope_quality)
      upper_and_lower = Both(upper,lower)
      if "test" in line:
        processed_data[index] = RunData(upper_and_lower, None)
      else:
        test_data = processed_data[index].test
        processed_data[index] = RunData(test_data, upper_and_lower)

  return processed_data, train_metric, train_cg_metric, train_upper_metric, train_lower_metric