import os
import numpy as np


# 날짜, 시간 관련 문자열 형식
BASE_DIR = os.getcwd()

def sigmoid(x):
    x = max(min(x, 10), -10)
    return 1. / (1. + np.exp(-x))
