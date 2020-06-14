#!/usr/bin/env python
# coding: utf-8

import os
import time
import tempfile
import subprocess
import numpy as np

from z3 import *
from datetime import datetime
from sexpdata import loads, dumps


epsilon = 0.001

def verifier(monotonic_indices,path,layers,monotonicity_direction,min_dict,max_dict,variable_types,model="",logging="",isUpper = True, prefix_path=""):
    s = Optimize()
    variableMap ={}
    variableMapF2 ={}
    input_features = len(variable_types)
    variableNameForIndex = ""
    for i in range(0,input_features):
        if variable_types[i] == "Real":
            variableMap[i] = Real('x'+str(i))
            variableMapF2[i] = Real('y'+str(i))
        else:
            if variable_types[i] == "Int":
                variableMap[i] = Int('x'+str(i))
                variableMapF2[i] = Int('y'+str(i))

    directory = path
    weightDict ={}
    biasDict = {}
    for i in range(0,layers):
        weightDict[i] = np.loadtxt(open(directory+"weights_layer"+str(i)+".csv", "rb"), delimiter=",")
        biasDict[i] = np.loadtxt(open(directory+"bias_layer"+str(i)+".csv", "rb"), delimiter=",")

    NN1 = nn_encoding(variableMap,weightDict,"x",layers,biasDict,s)
    NN2 = nn_encoding(variableMapF2,weightDict,"y",layers,biasDict,s)
    
    for index in range(0,len(variable_types)):
        if variable_types[index] == "Real":
            s.add(variableMapF2[index] >= RealVal(min_dict[index]))
            s.add(variableMapF2[index] <= RealVal(max_dict[index]))
            s.add(variableMap[index] >= RealVal(min_dict[index]))
            s.add(variableMap[index] <= RealVal(max_dict[index]))
        else:
            s.add(variableMapF2[index] >= IntVal(min_dict[index]))
            s.add(variableMapF2[index] <= IntVal(max_dict[index]))
            s.add(variableMap[index] >= IntVal(min_dict[index]))
            s.add(variableMap[index] <= IntVal(max_dict[index]))

    monotonicity_direction = int(monotonicity_direction)
    if monotonicity_direction == 0:
        for monotonic_index in monotonic_indices:
            s.add(variableMapF2[monotonic_index]<variableMap[monotonic_index])
            s.add(variableMap[monotonic_index] - variableMapF2[monotonic_index] > epsilon)
        s.add(NN2 - NN1 > epsilon)
        s.maximize(NN2-NN1)
    else:
        for monotonic_index in monotonic_indices:
            s.add(variableMapF2[monotonic_index]>variableMap[monotonic_index])
            s.add(variableMapF2[monotonic_index] - variableMap[monotonic_index]> epsilon)
        s.add(NN2 -NN1 > epsilon)
        s.maximize(NN2-NN1)
    tf = tempfile.NamedTemporaryFile()
    with open(prefix_path+tf.name, mode='w') as f:
        f.write("(set-option :produce-models true)\n")
        sexpr = s.sexpr()
        sexpr = sexpr.replace(")\n(check-sat)","")
        sexpr = sexpr + "\n :id f_y )\n(check-sat)\n"
        f.write(sexpr)
        f.write("(exit)")
    start_time = datetime.now()
    try:
        optimatsatsolver_path = "optimathsat"
        cmd = optimatsatsolver_path+" "+prefix_path+tf.name
        p = subprocess.Popen([cmd, ""], shell=True,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, close_fds=True)
        p.wait()
        returned_value = p.stdout.read().decode('UTF-8')
        elapsed_time = datetime.now() - start_time
        if "unsat" in returned_value:
            return "UNSAT", None, elapsed_time.total_seconds()
        else:
            if "sat" in returned_value:
                return "SAT", returned_value, elapsed_time.total_seconds()
    except Exception as e:
        print("Exception ", e)
        return "Exception while verifying",None, None

def counter_pair_generator(datapoint, monotonic_index, index,label,path,layers,monotonicity_direction,min,max,variable_types,model="",logging="",isUpper = True, prefix_path=""):
    s = Optimize()
    variableMap ={}
    variableMapF2 ={}
    input_features = len(datapoint)
    variableNameForIndex = ""
    
    for i in range(0,input_features):
        if variable_types[i] == "Real":
            tmp = Real('x'+str(i))
            tmp = RealVal(datapoint[i])
        else:
            if variable_types[i] == "Int":
                tmp = Int('x'+str(i))
                tmp = IntVal(datapoint[i])
        variableMap[i] = tmp
        if i == monotonic_index:
            if variable_types[i] == "Real":
                variableMapF2[i] = Real('y'+str(i))
                variableMap[i] = Real('x'+str(i))
                variableNameForIndex = 'y'+str(i)
            else:
                if variable_types[i] == "Int":
                    variableMapF2[i] = Int('y'+str(i))
                    variableNameForIndex = 'y'+str(i)
                    variableMap[i] = Int('x'+str(i))
        else:
            variableMapF2[i] = variableMap[i]

    directory = path
    weightDict ={}
    biasDict = {}
    for i in range(0,layers):
        weightDict[i] = np.loadtxt(open(directory+"weights_layer"+str(i)+".csv", "rb"), delimiter=",")
        biasDict[i] = np.loadtxt(open(directory+"bias_layer"+str(i)+".csv", "rb"), delimiter=",")
    
    NN1 = nn_encoding(variableMap,weightDict,"x",layers,biasDict,s)
    NN2 = nn_encoding(variableMapF2,weightDict,"y",layers,biasDict,s)
        
    if variable_types[monotonic_index] == "Real":
        s.add(variableMapF2[monotonic_index] >= RealVal(min))
        s.add(variableMapF2[monotonic_index] <= RealVal(max))
        s.add(variableMap[monotonic_index] >= RealVal(min))
        s.add(variableMap[monotonic_index] <= RealVal(max))
    else:
        s.add(variableMapF2[monotonic_index] >= IntVal(min))
        s.add(variableMapF2[monotonic_index] <= IntVal(max))
        s.add(variableMap[monotonic_index] >= IntVal(min))
        s.add(variableMap[monotonic_index] <= IntVal(max))

    logging.debug("\nSolving with OptimathSat")
    logging.debug('x\' > %2f'%(float(min)))
    logging.debug('x\' < %2f'%(float(max)))
    logging.debug('x = %2f'%(datapoint[monotonic_index]))
    logging.debug('y = %2f'%(label))
    monotonicity_direction = int(monotonicity_direction)
    logging.debug("The type of monotonicity direction and the value is %s"%(monotonicity_direction))
    logging.debug(type(monotonicity_direction))

    if isUpper:
        if monotonicity_direction == 0:
            logging.debug("Increasing Monotonicity")
            logging.debug('x\' < %2f'%(datapoint[monotonic_index]))
            s.add(variableMapF2[monotonic_index]<variableMap[monotonic_index])
            s.add(variableMap[monotonic_index] - variableMapF2[monotonic_index] > epsilon)
            s.add(NN2 - NN1 > epsilon)
            s.maximize(NN2-NN1)
        else:
            logging.debug("Decreasing Monotonicity")
            logging.debug('y > %2f'%(datapoint[monotonic_index]))
            s.add(variableMapF2[monotonic_index]>variableMap[monotonic_index])
            s.add(variableMapF2[monotonic_index] - variableMap[monotonic_index]> epsilon)
            s.add(NN2 - NN1 > epsilon)
            s.maximize(NN2-NN1)
    tf = tempfile.NamedTemporaryFile()
    with open(prefix_path+tf.name, mode='w') as f:
        f.write("(set-option :produce-models true)\n")
        # f.write("(set-option :timeout 20.0)\n")
        # f.write("(set-option :config opt.soft_timeout=true)\n")

        sexpr = s.sexpr()
        sexpr = sexpr.replace(")\n(check-sat)","")
        sexpr = sexpr + "\n :id diff )\n(check-sat)"
        f.write(sexpr)
        # f.write("(get-value ("+variableNameForIndex+"))\n(get-value (f_y))\n(exit)")
        f.write("(get-value (x"+str(monotonic_index)+"))\n(get-value ("+variableNameForIndex+"))\n(get-value (diff))\n(exit)")
    start_time = time.time()
    try:
        optimatsatsolver_path = "optimathsat"
        cmd = optimatsatsolver_path+" "+prefix_path+tf.name
        elapsed_time = time.time() - start_time
        p = subprocess.Popen([cmd, ""], shell=True,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, close_fds=True)
        p.wait()
        elapsed_time = time.time() - start_time
        # subprocess.call(cmd,shell = True) 
        returned_value = p.stdout.read().decode('UTF-8')
        if "unsat" in returned_value:
            logging.debug("Unsat: No model")
            return None, elapsed_time,index
        returned_values  = returned_value.splitlines()
        solved_vals = []
        for v in returned_values:
            if "sat" in v:
                continue
            name, val = parseSexp(v)
            if 'x' in dumps(name) or 'y' in dumps(name):
                solved_vals.append(val)
        return solved_vals,elapsed_time,index
    except Exception as e:
        print("Exception ", e)
        elapsed_time = time.time() - start_time
        return None,elapsed_time,index


def nn_encoding(variableMap, weightDict, type_nn,layers,biasDict,s):
    input_vars = variableMap
    hiddenLayerConstraintMapYF2 = {}
    hiddenLayerConstraintMapXF2 = {}
    for i in range(0,layers-1):
        z3LayerConstraintsFL1 = generateReluConstraintsEachLayer(input_vars,weightDict[i],biasDict[i],i,hiddenLayerConstraintMapYF2,hiddenLayerConstraintMapXF2,type_nn,s)
        input_vars = hiddenLayerConstraintMapXF2
        hiddenLayerConstraintMapYF2 = {}
        hiddenLayerConstraintMapXF2 = {}
    i = i + 1
    encoding = Real("finallayer_"+type_nn)
    encoding = generateFinalLayerConstraint(input_vars,weightDict[i],biasDict[i])
    return encoding


def counter_example_generator_upper_env(datapoint, monotonic_indices, label, path, layers, monotonicity_direction, min_dict, max_dict, variable_types, logging="", prefix_path="", isClassification = False):
    s = Optimize()
    variableMap ={}
    variableMapF2 ={}
    input_features = len(datapoint)
    variableNameForIndex = ""
    for i in range(0,input_features):
        if variable_types[i] == "Real":
            tmp = Real('x'+str(i))
            tmp = RealVal(datapoint[i])
        else:
            if variable_types[i] == "Int":
                tmp = Int('x'+str(i))
                tmp = IntVal(datapoint[i])
        variableMap[i] = tmp
        if i in monotonic_indices:
            if variable_types[i] == "Real":
                variableMapF2[i] = Real('y'+str(i))
                variableNameForIndex = 'y'+str(i)
            else:
                if variable_types[i] == "Int":
                    variableMapF2[i] = Int('y'+str(i))
                    variableNameForIndex = 'y'+str(i)
        else:
            variableMapF2[i] = variableMap[i]

    directory = path
    weightDict ={}
    biasDict = {}
    for i in range(0,layers):
        weightDict[i] = np.loadtxt(open(directory+"weights_layer"+str(i)+".csv", "rb"), delimiter=",")
        biasDict[i] = np.loadtxt(open(directory+"bias_layer"+str(i)+".csv", "rb"), delimiter=",")

    NN1 = nn_encoding(variableMap,weightDict,"x",layers,biasDict,s)
    NN2 = nn_encoding(variableMapF2,weightDict,"y",layers,biasDict,s)
    
    for monotonic_index in monotonic_indices:
        if variable_types[monotonic_index] == "Real":
            s.add(variableMapF2[monotonic_index] >= RealVal(min_dict[monotonic_index]))
            s.add(variableMapF2[monotonic_index] <= RealVal(max_dict[monotonic_index]))
        else:
            s.add(variableMapF2[monotonic_index] >= IntVal(min_dict[monotonic_index]))
            s.add(variableMapF2[monotonic_index] <= IntVal(max_dict[monotonic_index]))

    monotonicity_direction = int(monotonicity_direction)
    logging.debug("The type of monotonicity direction and the value is %s"%(monotonicity_direction))
    logging.debug(type(monotonicity_direction))
    if monotonicity_direction == 0:
        logging.debug("Increasing Monotonicity")
        for monotonic_index in monotonic_indices:
            logging.debug('x\' < %2f'%(datapoint[monotonic_index]))
            s.add(variableMapF2[monotonic_index]<=variableMap[monotonic_index])
            # s.add(variableMap[monotonic_index] - variableMapF2[monotonic_index] > epsilon)
        if isClassification:
            s.add(NN2 - NN1 > epsilon)
            s.maximize(NN2-NN1)
        else:
            s.add(NN2 - label > epsilon)
            s.maximize(NN2-label)
    else:
        logging.debug("Decreasing Monotonicity")
        for monotonic_index in monotonic_indices:
            # logging.debug('y > %2f'%(datapoint[monotonic_index]))
            s.add(variableMapF2[monotonic_index]>=variableMap[monotonic_index])
            # s.add(variableMapF2[monotonic_index] - variableMap[monotonic_index]> epsilon)
        if isClassification:
            s.add(NN2 - NN1 > epsilon)
            s.maximize(NN2-NN1)
        else:
            s.add(NN2 -label > epsilon)
            s.maximize(NN2-label)
    # print("The number of constraints in upper verifier is " + str(len(s.assertions())))
    tf = tempfile.NamedTemporaryFile()
    with open(prefix_path+tf.name, mode='w') as f:
        f.write("(set-option :produce-models true)\n")
        sexpr = s.sexpr()
        sexpr = sexpr.replace(")\n(check-sat)","")
        sexpr = sexpr + "\n :id f_y )\n(check-sat)\n"
        f.write(sexpr)
        for i in monotonic_indices:
            variableNameForIndex = 'y'+str(i)
            f.write("(get-value ("+variableNameForIndex+"))\n")
        f.write("(get-value (f_y))\n(exit)")

    return solve(logging,variable_types,monotonic_indices,input_features,datapoint, tf,prefix_path)

def counter_example_generator_lower_env(datapoint, monotonic_indices, label, path, layers, monotonicity_direction, min_dict, max_dict, variable_types, logging="", prefix_path="", isClassification = False):
    s = Optimize()
    variableMap ={}
    variableMapF2 ={}
    input_features = len(datapoint)
    variableNameForIndex = ""
    for i in range(0,input_features):
        if variable_types[i] == "Real":
            tmp = Real('x'+str(i))
            tmp = RealVal(datapoint[i])
        else:
            if variable_types[i] == "Int":
                tmp = Int('x'+str(i))
                tmp = IntVal(datapoint[i])
        variableMap[i] = tmp
        if i in monotonic_indices:
            if variable_types[i] == "Real":
                variableMapF2[i] = Real('y'+str(i))
                variableNameForIndex = 'y'+str(i)
            else:
                if variable_types[i] == "Int":
                    variableMapF2[i] = Int('y'+str(i))
                    variableNameForIndex = 'y'+str(i)
        else:
            variableMapF2[i] = variableMap[i]
        
    directory = path
    weightDict ={}
    biasDict = {}
    for i in range(0,layers):
        weightDict[i] = np.loadtxt(open(directory+"weights_layer"+str(i)+".csv", "rb"), delimiter=",")
        biasDict[i] = np.loadtxt(open(directory+"bias_layer"+str(i)+".csv", "rb"), delimiter=",")

    NN1 = nn_encoding(variableMap,weightDict,"x",layers,biasDict,s)
    NN2 = nn_encoding(variableMapF2,weightDict,"y",layers,biasDict,s)

    for monotonic_index in monotonic_indices:
        if variable_types[monotonic_index] == "Real":
            s.add(variableMapF2[monotonic_index] >= RealVal(min_dict[monotonic_index]))
            s.add(variableMapF2[monotonic_index] <= RealVal(max_dict[monotonic_index]))
        else:
            s.add(variableMapF2[monotonic_index] >= IntVal(min_dict[monotonic_index]))
            s.add(variableMapF2[monotonic_index] <= IntVal(max_dict[monotonic_index]))

    monotonicity_direction = int(monotonicity_direction)
    logging.debug("The type of monotonicity direction and the value is %s"%(monotonicity_direction))
    logging.debug(type(monotonicity_direction))
    if monotonicity_direction == 0:
        logging.debug("Increasing Monotonicity")
        for monotonic_index in monotonic_indices:
            logging.debug('y < %2f'%(datapoint[monotonic_index]))
            s.add(variableMapF2[monotonic_index] >= variableMap[monotonic_index])
            # s.add(variableMapF2[monotonic_index] - variableMap[monotonic_index] > epsilon )
        if isClassification:
            s.add(NN1 - NN2 > epsilon)
            s.maximize(NN1 - NN2)
        else:
            s.add(label - NN2 > epsilon)
            s.maximize(label - NN2)
    else:
        logging.debug("Decreasing Monotonicity")
        for monotonic_index in monotonic_indices:
            # logging.debug('y > %2f'%(datapoint[monotonic_index]))
            s.add(variableMapF2[monotonic_index] <= variableMap[monotonic_index])
            # s.add(variableMap[monotonic_index] - variableMapF2[monotonic_index] > epsilon )
        if isClassification:
            s.add(NN1 - NN2 > epsilon)
            s.maximize(NN1 - NN2)
        else:
            s.add(label - NN2 > epsilon)
            s.maximize(label - NN2)
    # print("The number of constraints in lower verifier is " + str(len(s.assertions())))
    tf = tempfile.NamedTemporaryFile()
    with open(prefix_path+tf.name, mode='w') as f:
        f.write("(set-option :produce-models true)\n")
        sexpr = s.sexpr()
        sexpr = sexpr.replace(")\n(check-sat)","")
        sexpr = sexpr + "\n :id f_y )\n(check-sat)\n"
        f.write(sexpr)
        for i in monotonic_indices:
            variableNameForIndex = 'y'+str(i)
            f.write("(get-value ("+variableNameForIndex+"))\n")
        f.write("(get-value (f_y))\n(exit)")

    return solve(logging, variable_types, monotonic_indices, input_features, datapoint, tf, prefix_path)

def parseSexp(sexp):
    parsed_sexp = loads(sexp)
    val = 0.0
    if '/' not in sexp:
        val = int(parsed_sexp[0][1])
    else:
        if '-' in sexp:
            val = -1 * float((1.0*parsed_sexp[0][1][1][1])/parsed_sexp[0][1][1][2])
        else:
            val = float((1.0*parsed_sexp[0][1][1])/parsed_sexp[0][1][2])
    return (parsed_sexp[0][0],val)


def solve(logging, variable_types,monotonic_indices,input_features,datapoint, smtFileName = "",prefix_path=""):
    start_time = time.time()
    try:
        optimatsatsolver_path = "optimathsat"
        cmd = optimatsatsolver_path+" "+ prefix_path +smtFileName.name

        elapsed_time = time.time() - start_time
        p = subprocess.Popen([cmd, ""], shell=True,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, close_fds=True)
        p.wait()
        elapsed_time = time.time() - start_time
        counter_examples = []
        returned_value = p.stdout.read().decode('UTF-8')
        if "unsat" in returned_value:
            logging.debug("Unsat: No model")
            return None, elapsed_time, None
        
        noOfLines = len(returned_value.splitlines())
        f_y_parsedsexp = loads(returned_value.splitlines()[noOfLines-1])

        f_y = 0.0
        if '-' in returned_value.splitlines()[noOfLines-1]:
            f_y = -1 * float((1.0*f_y_parsedsexp[0][1][1][1])/f_y_parsedsexp[0][1][1][2])
        else:
            f_y = float((1.0*f_y_parsedsexp[0][1][1])/f_y_parsedsexp[0][1][2])

        cg_parsedsexp = {}
        count = 1
        for monotonic_index in monotonic_indices:
            cg_parsed = loads(returned_value.splitlines()[count])
            cg = 0
            if variable_types[monotonic_index] == "Int":
                cg = int(cg_parsed[0][1])
            else:
                if '/' in returned_value.splitlines()[count]:
                    if '-' in returned_value.splitlines()[count]:
                        cg = -1 * float(cg_parsed[0][1][1][1]/cg_parsed[0][1][1][2])
                    else:
                        cg = float(cg_parsed[0][1][1]/cg_parsed[0][1][2])
                else:
                    cg = float(cg_parsed[0][1])
            count = count + 1
            cg_parsedsexp[monotonic_index] = cg

        logging.debug('violation = %2f'%(f_y))
        for i in range(0,input_features):
            if i in monotonic_indices:
                datapoint[i] = cg_parsedsexp[i]
        os.remove(prefix_path +smtFileName.name)
        return datapoint,elapsed_time,f_y
    except Exception as e:
        print("Exception "+ str(e))
        elapsed_time = time.time() - start_time
        os.remove(prefix_path +smtFileName.name)
        return None,elapsed_time,None

def generateFinalLayerConstraint(z3LayerConstraints,weights,bias):
    count =0
    constraints =[]
    for i in weights:
        prod = z3.Product(z3LayerConstraints[count],i)
        if(len(constraints)>0):
            constraint=z3.Sum(constraints.pop(),prod)
            constraints.append(constraint)
        else:
            constraints.append(prod)
        count = count+1
    return z3.Sum(constraint,bias)

def generateReluConstraintsEachLayer(variableMap, weights, bias,layerNo,hiddenLayerConstraintMapInput,hiddenLayerConstraintMapOutput,subscript,s):
    outercount=0
    z3LayerConstraints = {}
    constraints = []
    for i in np.transpose(weights):
        count = 0
        constraint = None
        constraints =[]
        for j in i:
            if isinstance(variableMap[count],z3.z3.IntNumRef) or (isinstance(variableMap[count],z3.z3.ArithRef) and variableMap[count].is_int()):
                prod = z3.Product(ToReal(variableMap[count]),j)
            else:
                prod = z3.Product(variableMap[count],j)

            if(len(constraints)>0):
                constraint=z3.Sum(constraints.pop(),prod)
                constraints.append(constraint)
            else:
                constraints.append(prod)
            count = count+1            
        constraint = z3.Sum(constraint,bias[outercount])
        hiddenLayerConstraintMapInput[outercount] = Real("layer_"+str(layerNo)+"hiddeninput_"+subscript+str(layerNo)+str(outercount))
        hiddenLayerConstraintMapInput[outercount] = constraint
        hiddenLayerConstraintMapOutput[outercount] = Real("layer"+str(layerNo)+"hiddenoutput_"+subscript+str(layerNo)+str(outercount))
        s.add(Implies(hiddenLayerConstraintMapInput[outercount]>0,hiddenLayerConstraintMapOutput[outercount]>=hiddenLayerConstraintMapInput[outercount]))
        s.add(Implies(hiddenLayerConstraintMapInput[outercount]>0,hiddenLayerConstraintMapOutput[outercount]<=hiddenLayerConstraintMapInput[outercount]))
        s.add(Implies(hiddenLayerConstraintMapInput[outercount]<=0,hiddenLayerConstraintMapOutput[outercount]>=0))
        s.add(Implies(hiddenLayerConstraintMapInput[outercount]<=0,hiddenLayerConstraintMapOutput[outercount]<=0))
        z3LayerConstraints[outercount]=(constraint)
        outercount = outercount+1
    return z3LayerConstraints
