from hex.player import Player
from hex.game import Game

import random
import numpy as np

class ModelPlayer(Player):
    
    def __init__(self, hex_model):
        """Create a model based player, with a given model"""
        self.hex_model = hex_model
        self.stats = []
        
    def reset_stats(self):
        self.stats = []
    
    def move(self, board):
        possibleMoves = board.getPossibleMoves()
        random.shuffle(possibleMoves)
        
        converted_board = self.canonicalize_board(board)
        if possibleMoves:
            possible_moves = [(pos, dec(nice_pos_to_loc(pos))) for pos in possibleMoves]
            boards = np.array([apply_move(converted_board, move, self.role) for _, move in possible_moves ])
            predictions = self.hex_model.predict(boards)

            chosen_move = possible_moves[np.argmax(predictions)][0]
            
            self.stats.append(predictions)

            return chosen_move
        else:
            raise Exception("No possible moves to make.")
            
    def canonicalize_board(self, board):
        given_board = board.getStateForPlayer(BLACK if self.role == "black" else WHITE)
        
        if self.role == "black":
            black_board = (given_board == 1).astype(np.int8)
            white_board = (given_board == -1).astype(np.int8)
        else:
            white_board = (given_board == 1).astype(np.int8)
            black_board = (given_board == -1).astype(np.int8)

        final_board = np.stack([black_board, white_board], axis=-1)
        
        return final_board
    

def apply_move(board, move, player):
    board = board.copy()
    x, y = move
    player_idx = (BLACK - 1) if player == "black" else (WHITE - 1)
    
    board[x, y, player_idx] = 1
    
    return board

def nice_pos_to_loc(pos_pair):
    return (pos_pair[0], ord(pos_pair[1].lower()) - 96)
    
def dec(t):
    return (t[0] - 1, t[1] - 1)

BLACK = 1
WHITE = 2

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Input
from tensorflow.keras.models import Model

WEIGHTS_PATH = "olivia_model_final_weights"
BATCH_SIZE = 1000
NUM_EPOCHS = 10
#STEPS_PER_EPOCH = 30
START_LEARNING_RATE = 0.0025

class OliviaModel(object):

    def __init__(self):
        # a layer instance is callable on a tensor, and returns a tensor
        self.model = tf.keras.Sequential([
            Conv2D(32, kernel_size=(2, 2), strides=(1, 1),
                          data_format='channels_last',
                          activation='relu',
                          input_shape=(13, 13, 2)),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPool2D(pool_size=(2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        self.model.summary()

        self.model.compile(optimizer=tf.train.AdamOptimizer(START_LEARNING_RATE),
                  loss='mse',       # mean squared error
                  metrics=['mae'])  # mean absolute error
        
        self.load(WEIGHTS_PATH)

    def predict(self, boards, **kwargs):
        """Given a board predict the winner using as a 2-vector [black_win_prob, white_win_prob]"""
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

player = ModelPlayer(OliviaModel())