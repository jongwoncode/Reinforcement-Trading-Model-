import os
import time
import threading
import random
import numpy as np
import tensorflow as tf

from tensorflow.compat.v1.train import AdamOptimizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



# ActorCritic 인공신경망
class LSTM_DNN_AC(tf.keras.Model) :
    def __init__(self, action_size, n_steps, state_size, balance_size) :
        super(LSTM_DNN_AC, self).__init__()
        self.action_size = action_size
        self.n_steps = n_steps
        self.state_size = state_size
        self.balance_size = balance_size

        # chart_state : (n_steps, chart_columns) ex) (5, 28)
        self.lstm1 = LSTM(128, activation='tanh', dropout= 0.3, return_sequences=True)
        self.lstm2 = LSTM(64, activation = 'tanh', dropout = 0.3, return_sequences=True)
        self.lstm3 = LSTM(16, activation = 'tanh', return_sequences=False)
        # balance_state : (balance_info) ex) (4)
        self.dnn1 = Dense(64, activation='relu')
        self.dnn2 = Dense(16, activation='relu')
        # concatenate & shared network
        self.concatenate = Concatenate()
        self.shared_fc = Dense(128, activation='relu')
        # Actor part
        self.poilcy1 = Dense(32, activation='relu')
        self.policy = Dense(action_size, activation='linear')
        # Critic part
        self.value1 = Dense(32, activation='relu')
        self.value = Dense(1, activation='linear')
    
    def call(self, inputs) :
        # LSTM PART
        c_inp = inputs[0]
        c = self.lstm1(c_inp)
        c = self.lstm2(c)
        c = self.lstm3(c)
        # DNN PART
        b_inp = inputs[1]
        b = self.dnn1(b_inp)
        b = self.dnn2(b)
        # CONCATENATE & SHARED PART -> (...chart, ...balance) (16+16, )
        shared = self.concatenate([c, b])
        shared = self.shared_fc(shared)
        # ACTOR PART
        policy = self.poilcy1(shared)
        policy = self.policy(policy)
        # CRITIC PART
        value = self.value1(shared)
        value = self.value(value)
        return policy, value

    # Subclassing API에서 plot_model을 그리기 위해서 build_graph 메서드 생성
    def build_graph(self) :
        c_inp = Input(shape= (self.n_steps, self.state_size), dtype=tf.float32)
        b_inp = Input(shape= (self.balance_size, ), dtype=tf.float32)
        return Model(inputs=[c_inp, b_inp], outputs=self.call([c_inp, b_inp]))

