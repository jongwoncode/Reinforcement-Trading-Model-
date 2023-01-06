import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Concatenate, BatchNormalization, Dropout

# ActorCritic 인공신경망
class LSTM_DNN_AC(tf.keras.Model) :
    def __init__(self, action_size, n_steps, state_size, balance_size) :
        super(LSTM_DNN_AC, self).__init__()
        self.action_size = action_size
        self.n_steps = n_steps
        self.state_size = state_size
        self.balance_size = balance_size

        # chart_state : (n_steps, chart_columns) ex) (5, 28)
        self.lstm1 = LSTM(128, activation='tanh', dropout= 0.4, return_sequences=True)
        self.lstm2 = LSTM(64, activation = 'tanh',dropout= 0.4, return_sequences=True)
        self.lstm3 = LSTM(64, activation = 'tanh',dropout= 0.4, return_sequences=False)
        self.batch_lstm3 = BatchNormalization()

        # balance_state : (balance_info) ex) (4)
        self.dnn1 = Dense(128, activation='relu')
        self.drop_dnn1 = Dropout(0.4)
        self.batch_dnn1 = BatchNormalization()

        self.dnn2 = Dense(64, activation='relu')
        self.drop_dnn2 = Dropout(0.4)
        self.batch_dnn2 = BatchNormalization()

        # concatenate & shared network
        self.concatenate = Concatenate()
        self.shared_fc = Dense(64, activation='relu')
        self.drop_shared = Dropout(0.4)
        self.batch_shared = BatchNormalization()

        # Actor part & Critic part
        self.policy1 = Dense(32, activation = 'relu')
        self.policy = Dense(action_size, activation='linear')

        self.value1 = Dense(32, activation = 'relu')
        self.value = Dense(1, activation='linear')
    
    def call(self, inputs) :
        # LSTM PART & BatchNormalization(only Last part)
        c_inp = inputs[0]
        c = self.lstm1(c_inp)
        c = self.lstm2(c)
        c = self.lstm3(c)
        c = self.batch_lstm3(c)

        # DNN PART $ BatchNormalization
        b_inp = inputs[1]
        b = self.dnn1(b_inp)
        b = self.drop_dnn1(b)
        b = self.batch_dnn1(b)

        b = self.dnn2(b)
        b = self.drop_dnn2(b)
        b = self.batch_dnn2(b)

        # CONCATENATE & SHARED PART -> (...chart, ...balance) (16+16, ) & BatchNormalization
        shared = self.concatenate([c, b])
        shared = self.shared_fc(shared)
        shared = self.drop_shared(shared)
        shared = self.batch_shared(shared)
        # ACTOR PART
        policy = self.policy1(shared)
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



class DNN_AC(tf.keras.Model) :
    def __init__(self, action_size, state_size, balance_size) :
        super(DNN_AC, self).__init__()
        self.action_size = action_size
        self.state_size = state_size
        self.balance_size = balance_size
        # gather input
        self.concatenate = Concatenate()
    
        # Share Network : first layer
        self.dnn1 = Dense(128, activation='relu')
        self.drop_dnn1 = Dropout(0.4)
        self.batch_dnn1 = BatchNormalization()

        # Share Network : Second layer
        self.dnn2 = Dense(64, activation='relu')
        self.drop_dnn2 = Dropout(0.4)
        self.batch_dnn2 = BatchNormalization()

        # Actor part & Critic part
        self.policy1 = Dense(32, activation = 'relu')
        self.policy = Dense(action_size, activation='linear')

        self.value1 = Dense(32, activation = 'relu')
        self.value = Dense(1, activation='linear')
    
    def call(self, inputs) :
        inp = self.concatenate([inputs[0], inputs[1]])
        # SHARE PART
        shared = self.dnn1(inp)
        shared = self.drop_dnn1(shared)
        shared = self.batch_dnn1(shared)
        shared = self.dnn2(shared)
        shared = self.drop_dnn2(shared)
        shared = self.batch_dnn2(shared)

        # ACTOR PART
        policy = self.policy1(shared)
        policy = self.policy(policy)

        # CRITIC PART
        value = self.value1(shared)
        value = self.value(value)
        return policy, value

    # Subclassing API에서 plot_model을 그리기 위해서 build_graph 메서드 생성
    def build_graph(self) :
        c_inp = Input(shape= (self.state_size, ), dtype=tf.float32)
        b_inp = Input(shape= (self.balance_size, ), dtype=tf.float32)
        return Model(inputs=[c_inp, b_inp], outputs=self.call([c_inp, b_inp]))