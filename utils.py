import time
import datetime
import os
import numpy as np

BASE_DIR = os.getcwd()
LOGGER_NAME = 'rltrader'
FORMAT_DATETIME = "%Y%m%d%H%M"

def get_time_str():
    return datetime.datetime.fromtimestamp(
        int(time.time())).strftime(FORMAT_DATETIME)

def softmax(x) : 
    c = np.max(x) 
    exp_a = np.exp(x-c) 
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y