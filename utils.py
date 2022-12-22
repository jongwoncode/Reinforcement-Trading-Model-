import os
import numpy as np

BASE_DIR = os.getcwd()
LOGGER_NAME = 'rltrader'


def sigmoid(x):
    x = max(min(x, 10), -10)
    return 1. / (1. + np.exp(-x))

'''https://3months.tistory.com/66 '''
def softmax(a) : 
    c = np.max(a) # 최댓값
    exp_a = np.exp(a-c) # 각각의 원소에 최댓값을 뺀 값에 exp를 취한다. (이를 통해 overflow 방지)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y