#Take on the the real data file and modify it so there is 50& positives/negatives

import numpy as np
from numpy import genfromtxt
import pandas as pd 
from training import parameters as param

def run_data_process():
    trainMatrix = []
    real_data = genfromtxt(param.real_filename, delimiter=',')
    for row in real_data:
        if(row[param.features_size+param.labels_size-1]==1):
            trainMatrix.append(row)
    num_posi_values = len(trainMatrix)
    num_nega_values = 0
    i = 0
    while (num_nega_values!=num_posi_values):
        if(real_data[i][param.features_size+param.labels_size-1]==0):
            trainMatrix.append(real_data[i])
            num_nega_values = num_nega_values + 1
        i = i + 1
    trainMatrix = np.array(trainMatrix)
    np.savetxt(param.pessimist_filename, trainMatrix, delimiter=',', fmt="%d")
