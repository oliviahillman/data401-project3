#!/usr/bin/env python

from player import *

from hex.player import HumanPlayer
from collections import Counter
import random
from models.declan import DeclanModel
from models.natalie import NatalieModel
from player import ModelPlayer, convert_game_fixed_to_relative
from display import animate_board_choices

from tensorflow.train import checkpoint_exists

import os

BLACK = 1
WHITE = 2

def run():
    declan_model = DeclanModel()
    if checkpoint_exists('checkpoints/declan_model_weights'):
        declan_model.load('checkpoints/declan_model_weights')

    natalie_model = NatalieModel()

    declan_player = ModelPlayer(declan_model, relative=True)
    natalie_player = ModelPlayer(natalie_model)
    random_player = HumanPlayer()

    new_boards, new_winners, final_winner = play_game(declan_player,
                                                      random_player,
                                                      verbose=True,
                                                      relative=False)

if __name__ == '__main__':
    run()

