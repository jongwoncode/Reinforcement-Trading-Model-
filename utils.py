import time
import datetime
import os
import numpy as np

BASE_DIR = os.getcwd()
LOGGER_NAME = 'rltrader'
TEST_LOGGER_NAME = 'rltrader_test'

# 날짜, 시간 관련 문자열 형식
FORMAT_DATE = "%Y%m%d"
FORMAT_DATETIME = "%Y%m%d%H%M"

def get_today_str():
    today = datetime.datetime.combine(
        datetime.date.today(), datetime.datetime.min.time())
    today_str = today.strftime('%Y%m%d')
    return today_str

def get_time_str():
    return datetime.datetime.fromtimestamp(
        int(time.time())).strftime(FORMAT_DATETIME)

def softmax(x) : 
    c = np.max(x) 
    exp_a = np.exp(x-c) 
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y