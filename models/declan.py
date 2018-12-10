import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Input, BatchNormalization, LeakyReLU, add, GaussianNoise
from tensorflow.keras.models import Model
from . import HexModel

START_LEARNING_RATE = 1e-3
RESID_LAYERS = 1
NOISE_STDDEV = 2
NUM_FILTERS = 32

BLACK = 1
WHITE = 2

class DeclanModel(HexModel):

    def __init__(self):
        # a layer instance is callable on a tensor, and returns a tensor
        inputs = Input(shape=(13, 13, 3))
        
        x = Conv2D(NUM_FILTERS,
               kernel_size=(1, 1),
               strides=(1,1),
               activation=None,
               padding='same')(inputs)

        for r in range(RESID_LAYERS):
            x = residual_layer(x)
        
        x = BatchNormalization()(x)
        x = GaussianNoise(NOISE_STDDEV)(x)
        x = Flatten()(x)
        x = Dense(32, activation='relu')(x)
        x = Dense(1, activation='sigmoid')(x)
        
        self.model = Model(inputs=inputs, outputs=x)
        self.model.summary()

        self.model.compile(
            optimizer=tf.train.RMSPropOptimizer(START_LEARNING_RATE),
            loss='mse',       # mean squared error
            metrics=['mae'])  # mean absolute error

    def predict(self, boards, **kwargs):
        return self.model.predict(boards, **kwargs)

    def fit(self, boards, winners, *args, **kwargs):
        """Given a stack of boards and a winner of a game, train the model"""
        self.model.fit(boards, winners, *args, **kwargs)

    def save(self, filename):
        """Serialize model to disk"""
        self.model.save_weights(filename)

    def load(self, filename):
        """De-serialize model from disk"""
        self.model.load_weights(filename)

def residual_layer(in_x):
    x = Conv2D(NUM_FILTERS,
               kernel_size=(3, 3),
               strides=(1,1),
               activation=None,
               padding='same')(in_x)    
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(NUM_FILTERS,
               kernel_size=(3, 3),
               strides=(1,1),
               activation=None,
               padding='same')(x)
    x = BatchNormalization()(x)
    x = add([in_x, x])
    x = LeakyReLU()(x)
    
    return x
    