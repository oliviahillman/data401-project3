#!/usr/bin/env python

from player import *

NUM_EPOCHS = 4
REINFORCEMENT_ROUNDS = 16
REINFORCEMENT_BATCH_SIZE = 128
RELATIVE = True
TRAIN = True
        
from hex.player import RandomPlayer
from collections import Counter
import random
from models.declan import DeclanModel
from player import ModelPlayer
from display import animate_board_choices
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.train import checkpoint_exists

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def run_only_reinforce():
    
    callbacks = [
        ModelCheckpoint('checkpoints/declan_model_weights', monitor='val_loss', verbose=0,
                        save_best_only=True, save_weights_only=True, mode='auto', period=1)
    ]
    model = DeclanModel()
    
    if checkpoint_exists('checkpoints/declan_model_weights'):
        print("Loading from checkpoint")
        model.load('checkpoints/declan_model_weights')
    
    p1 = ModelPlayer(model, relative=RELATIVE)
    p2 = ModelPlayer(model, relative=RELATIVE)
    
    for r in range(REINFORCEMENT_ROUNDS):
        accumulated_boards = None
        accumulated_winners = None

        for i in range(REINFORCEMENT_BATCH_SIZE):
            switch = bool(random.getrandbits(1))
            verbose = (i == 0)

            if switch:
                p_black = p1
                p_white = p2
            else:
                p_black = p2
                p_white = p1

            new_boards, new_winners, final_winner = play_game(p_black, p_white,
                                                              verbose=verbose,
                                                              relative=RELATIVE)
            
            if accumulated_boards is None:
                accumulated_boards = new_boards
            else:
                accumulated_boards = np.concatenate((accumulated_boards, new_boards))
            
            if accumulated_winners is None:
                accumulated_winners = new_winners
            else:
                accumulated_winners = np.concatenate((accumulated_winners, new_winners))

            if verbose:
                print("Outputing {} {} animation".format(r, i))
                animate_board_choices('outputs/black-{}-{}.mp4'.format(r, i),
                                      p_black.stats, BLACK,
                                      relative=RELATIVE)
                animate_board_choices('outputs/white-{}-{}.mp4'.format(r, i),
                                      p_white.stats, WHITE,
                                      relative=RELATIVE)
                
            p1.reset_stats()
            p2.reset_stats()
        
        print(np.sum(accumulated_winners), '/', accumulated_winners.shape[0])
        print("Finished round {}".format(r))
        if TRAIN:
            model.fit(accumulated_boards, accumulated_winners,
                      epochs=NUM_EPOCHS, callbacks=callbacks,
                      validation_split=0.1)

def apply_move(board, move, player, value):
    x, y = move
    player_idx = (BLACK - 1) if player == "black" else (WHITE - 1)
    
    board[x, y, player_idx] = value
    
    return board

def nice_pos_to_loc(pos_pair):
    return (pos_pair[0], ord(pos_pair[1].lower()) - 96)
    
def dec(t):
    return (t[0] - 1, t[1] - 1)
    
run_only_reinforce()