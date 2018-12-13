#!/usr/bin/env python

from player import *

from hex.player import RandomPlayer, HumanPlayer
from collections import Counter
import random
from models.natalie_random import NatalieRandomModel
from models.olivia_random import OliviaRandomModel
from player import ModelPlayer, convert_game_fixed_to_relative
from display import animate_board_choices

import pandas as pd
from tensorflow.train import checkpoint_exists

import os
import itertools as it

BLACK = 1
WHITE = 2

NUM_EVAL_TRIALS = 250

def run():
    model_defs = [
        { 
            'name': 'natalie_random',
            'class': NatalieRandomModel,
            'weights': 'weights/natalie_model_random_weights'
        },
        {
            'name': 'olivia_random',
            'class': OliviaRandomModel,
            'weights': 'weights/olivia_model_random_weights'
        }
    ]

    random_player = RandomPlayer()
    
    for model_def in model_defs:
        model = model_def['class']()
        model.load(model_def['weights'])
        
        model_player = ModelPlayer(model)
        index = pd.MultiIndex.from_tuples(
            list(it.product(*[['random', model_def['name']], ['black', 'white']])),
            names=['model', 'color'])
        win_record = pd.Series(index=index, dtype=np.int16)
        for r in range(NUM_EVAL_TRIALS):
            switch = bool(random.getrandbits(1))
            if switch:
                p_black = model_player
                p_white = random_player
            else:
                p_black = random_player
                p_white = model_player

            _, _, win_color = play_game(p_black, p_white,
                                           verbose=False,
                                           relative=False)
            
            if switch:
                if win_color == 'black':
                    win_name = model_def['name']
                else:
                    win_name = 'random'
            else:
                if win_color == 'white':
                    win_name = model_def['name']
                else:
                    win_name = 'random'
            
            win_record[win_name, win_color] += 1
        
        print(win_record)
        print(win_record.groupby(level=0).sum())
        print(win_record.groupby(level=1).sum())
            
            
if __name__ == '__main__':
    run()

