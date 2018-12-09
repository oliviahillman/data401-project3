import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Input
from tensorflow.keras.models import Model
from . import HexModel

START_LEARNING_RATE = 0.0024787521766663585
NUM_EPOCHS = 10
BATCH_SIZE = 1000

class NatalieModel(HexModel):

    def __init__(self):
        # a layer instance is callable on a tensor, and returns a tensor
        self.model = tf.keras.Sequential([
            Conv2D(16,
                   kernel_size=(2,2),
                   strides=(1,1),
                   data_format='channels_last',
                   activation='relu',
                   input_shape=(13, 13, 3)),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(256, kernel_size=(3,3), strides=(1,1),
                          activation='relu'),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            Flatten(),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        self.model.summary()

        self.model.compile(optimizer=tf.train.AdamOptimizer(START_LEARNING_RATE),
                  loss='mse',       # mean squared error
                  metrics=['mae'])  # mean absolute error

    def predict(self, boards, **kwargs):
        """Given a board predict the winner using as a 2-vector [black_win_prob, white_win_prob]"""
        shape = boards.shape
        boards = np.array(
            [np.subtract(x[:,:,0], x[:,:,1]) for x in boards])
        boards = boards.reshape((shape[0], shape[1], shape[2], 1))
        boards_3 = np.tile(boards, 3)
        return self.model.predict(boards_3, **kwargs)

    def fit(self, boards, winners, *args, **kwargs):
        """Given a stack of boards and a winner of a game, train the model"""
        shape = boards.shape
        boards = np.array(
            [np.subtract(x[:,:,0], x[:,:,1]) for x in boards])
        boards = boards.reshape((shape[0], shape[1], shape[2], 1))
        boards_3 = np.tile(boards,3)
        self.model.fit(boards_3, winners, *args, **kwargs)

    def save(self, filename):
        """Serialize model to disk"""
        self.model.save_weights(filename)

    def load(self, filename):
        """De-serialize model from disk"""
        self.model.load_weights(filename)
