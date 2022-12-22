import os
import threading
import numpy as np

from tensorflow import *
from keras.models import Model
from keras.layers import Input, Dense, LSTM, BatchNormalization
from keras.optimizers import SGD, RMSprop

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        # INFO and WARNING messages are not printed


class LSTMNetwork :
    def __init__(self, input_dim=0, output_dim=0, shared_network=None, num_steps=1, lr=0.001, activation='linear', loss='mse') :
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_steps = num_steps
        self.lr = lr
        self.activation = activation
        self.loss = loss
        self.model = None
        self.shared_network = shared_network

        if self.shared_network is None :
            input = shared_network.input
            output = self.shared_network.output
        else :
            input = Input((self.input_dim))
            output = self.get_network_head(input).output

        output = Dense(self.output_dim, activation=self.activation, kernel_initializer='random_normal')(output)
        self.model = Model(input, output)
        self.model.compile(optimizer=SGD(learning_rate=self.lr), loss=self.loss)

    def predict(self, sample) :
        sample = np.array(sample).reshape((1, self.num_steps, self.input_dim))
        pred = self.model.predict_on_batch(sample).flatten()
        return pred
        
    def train_on_batch(self, x, y) :
        loss = 0.
        x = np.array(x).reshape((-1, self.num_stpes, self.input_dim))
        history = self.model.fit(x, y, epochs=10, verbose=False)
        loss += np.sum(history.history['loss'])
        return loss
    
    def save_model(self, model_path) :
        if model_path is not None and self.model is not None :
            self.model.save_weights(model_path, overwrite=True) 
        
    def load_model(self, model_path) :
        if model_path is not None :
            self.model.load_weights(model_path)

    def get_network_head(input) :
        output = LSTM(256, dropout=0.1, return_sequences=True, kernel_initializer='random_normal')(input)
        output = BatchNormalization()(output)
        output = LSTM(128, dropout=0.1, return_sequences=True, kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = LSTM(64, dropout=0.1, return_sequences=True, kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = LSTM(32, dropout=0.1, return_sequences=True, kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        return Model(input, output)
    

