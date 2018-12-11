import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Input, BatchNormalization, LeakyReLU, add
from tensorflow.keras.models import Model
from . import HexModel

START_LEARNING_RATE = 1e-4
RESID_LAYERS = 2
NOISE_STDDEV = 2
NUM_FILTERS = 8

BLACK = 1
WHITE = 2

class DeclanModel(HexModel):

    def __init__(self, exploration=0.25, dir_alpha=0.03):
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
        x = Flatten()(x)
        x = Dense(32, activation='relu')(x)
        x = Dense(1, activation='sigmoid')(x)
        
        self.model = Model(inputs=inputs, outputs=x)
        self.model.summary()

        self.model.compile(
            optimizer=tf.train.RMSPropOptimizer(START_LEARNING_RATE),
            loss='mse',       # mean squared error
            metrics=['mae'])  # mean absolute error
        
        self.exploration = exploration
        self.dir_alpha = dir_alpha

    def predict(self, boards, **kwargs):
        predictions = self.model.predict(boards, **kwargs)
        dir_alpha = np.full_like(predictions.squeeze(), self.dir_alpha)
        r1 = (1 - self.exploration) * predictions
        r2 = self.exploration * \
             np.random.dirichlet(dir_alpha)
        r = np.expand_dims(r1.squeeze() + r2.squeeze(), axis=-1)
        return r

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
    