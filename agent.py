import os
import time
import threading
import random
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1.train import AdamOptimizer

from network import *
from learner import *

# 브레이크아웃에서의 A3CAgent 클래스 (글로벌신경망)
class A3CAgent():
    def __init__(self, n_steps, chart_size, balance_size, action_size, chart_data=None, training_data=None, 
                initial_balance=100000000, min_trading_price=100000, max_trading_price=1000000):

        self.chart_data = chart_data 
        self.training_data = training_data
        self.initial_balance = initial_balance
        self.min_trading_price = min_trading_price
        self.max_trading_price = max_trading_price 
        self.action_size = action_size
        self.n_steps = n_steps
        self.chart_size = chart_size
        self.balance_size = balance_size
        # A3C 하이퍼파라미터
        self.discount_factor = 0.99
        self.no_op_steps = 30
        self.lr = 1e-4
        # 쓰레드의 갯수
        self.threads = 1
        # 전역 신경망 생성
        self.global_model = LSTM_DNN_AC(action_size, n_steps, chart_size, balance_size)
        # 전역 신경망 가중치 초기화 input : [chart_columns_shape, balance_state_shape]
        inputs = [tf.TensorShape((None, n_steps, chart_size)), tf.TensorShape((None, balance_size))]
        self.global_model.build(inputs)
        # 인공신경망 업데이트하는 옵티마이저 함수 생성
        self.optimizer = AdamOptimizer(self.lr, use_locking=True)
        # 텐서보드 설정 및 모델 가중치 저장 경로 설정
        self.writer = tf.summary.create_file_writer('output/rltrader')
        self.model_path = os.path.join(os.getcwd(), 'save_model', 'model')
        
    # 쓰레드를 만들어 학습을 하는 함수
    def train(self):
        # 쓰레드 수 만큼 Runner 클래스 생성
        runners = [Learner(self.chart_data, self.training_data, self.initial_balance, self.min_trading_price, 
                            self.max_trading_price, self.action_size, self.n_steps, self.chart_size, 
                            self.balance_size, self.global_model, self.optimizer, 
                            self.discount_factor, self.writer) for _ in range(self.threads)]
        # 각 쓰레드 시정
        for i, runner in enumerate(runners):
            print("Start worker #{:d}".format(i))
            runner.start()

        # 3분 (600초)에 한 번씩 모델을 저장
        while True:
            self.global_model.save_weights(self.model_path, save_format="tf")
            time.sleep(60 * 3)