#!/usr/bin/env python

from player import *
from models.basic import BasicHexModel

model = BasicHexModel()

BATCH_SIZE = 1000
NUM_EPOCHS = 10
STEPS_PER_EPOCH = 30

REINFORCEMENT_BATCH_SIZE = 1

SUBSET_SIZE = 10000
        
from hex.player import RandomPlayer
from collections import Counter
import random
from models.olivia import OliviaModel
from models.natalie import NatalieModel
from player import ModelPlayer
from display import animate_board_choices

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def run_only_reinforce():
    
    olivia_player = ModelPlayer(OliviaModel())
    natalie_player = ModelPlayer(NatalieModel())
    
    game_results = []

    for i in range(REINFORCEMENT_BATCH_SIZE):
        switch = bool(random.getrandbits(1))
        verbose = (i == 0)
        
        if switch:
            p_black = natalie_player
            p_white = olivia_player
        else:
            p_black = olivia_player
            p_white = natalie_player
        
        new_boards, new_winners, final_winner = play_game(p_black, p_white, verbose=verbose)
        
        if switch:
            if final_winner == 'black':
                game_results.append('natalie')
            else:
                game_results.append('olivia')
        else:
            if final_winner == 'black':
                game_results.append('olivia')
            else:
                game_results.append('natalie')
        
        if verbose:
            animate_board_choices('black-run-{}.mp4'.format(i), p_black.stats, BLACK)
            animate_board_choices('white-run-{}.mp4'.format(i), p_white.stats, WHITE)
        
    
    print(Counter(game_results))

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