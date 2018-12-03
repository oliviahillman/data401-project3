import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Input
from tensorflow.keras.models import Model
from . import HexModel

BATCH_SIZE = 32
NUM_EPOCHS = 10
STEPS_PER_EPOCH = 30
START_LEARNING_RATE = 0.01

class BasicHexModel(HexModel):
    
    def __init__(self):
        self.model = tf.keras.Sequential([
            Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                          data_format='channels_last',
                          activation='relu',
                          input_shape=(13, 13, 1)),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPool2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(2, activation='softmax')
        ])
        
        self.model.summary()
        
        self.model.compile(optimizer=tf.train.AdamOptimizer(START_LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['mae'])

    def predict(self, board, **kwargs):
        """Given a board predict the winner using as a 2-vector [black_win_prob, white_win_prob]"""
        return self.model.predict(np.array([board]), **kwargs)
        
    def train(self, boards, winners, *args, **kwargs):
        """Given a stack of boards and a winner of a game, train the model"""
        self.model.fit(boards, winners, *args, **kwargs)
        
    def save(self, fileanem):
        """Serialize model to disk"""
        model.save_weights(filename)
        
    def load(self, fileanem):
        """De-serialize model from disk"""
        model.load_weights(filename)