import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Input
from tensorflow.keras.models import Model
from . import HexModel

START_LEARNING_RATE = 0.003

class BasicHexModel(HexModel):
    
    def __init__(self):
        self.model = tf.keras.Sequential([
            Conv2D(128,
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   data_format='channels_last',
                   activation='relu',
                   padding='same',
                   input_shape=(13, 13, 2)),
            Conv2D(64,
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   activation='relu',
                   padding='same'),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPool2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        self.model.summary()
        
        self.model.compile(optimizer=tf.train.AdamOptimizer(START_LEARNING_RATE),
              loss='binary_crossentropy',
              metrics=['mae'])

    def predict(self, boards, **kwargs):
        """Given a board predict the winner using as a 2-vector [black_win_prob, white_win_prob]"""
        return self.model.predict(boards, **kwargs)
        
    def fit(self, boards, winners, *args, **kwargs):
        """Given a stack of boards and a winner of a game, train the model"""
        self.model.fit(boards, winners, *args, **kwargs)
        
    def save(self, fileanem):
        """Serialize model to disk"""
        model.save_weights(filename)
        
    def load(self, fileanem):
        """De-serialize model from disk"""
        model.load_weights(filename)