import time
import datetime
import os
import numpy as np
import tensorflow as tf

BASE_DIR = os.getcwd()
LOGGER_NAME = 'rltrader'
FORMAT_DATETIME = "%Y%m%d%H%M"

def get_time_str():
    return datetime.datetime.fromtimestamp(
        int(time.time())).strftime(FORMAT_DATETIME)

def softmax(x) : 
    c = tf.reduce_max(x) 
    exp_a = tf.math.exp(x-c) 
    sum_exp_a = tf.reduce_sum(exp_a)
    y = exp_a / sum_exp_a
    return y