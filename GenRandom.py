#!/usr/bin/env python

from player import *
from models.basic import BasicHexModel

model = BasicHexModel()

BATCH_SIZE = 1000
NUM_EPOCHS = 10
STEPS_PER_EPOCH = 30

REINFORCEMENT_BATCH_SIZE = 5
REPEATS = 3

SUBSET_SIZE = 10000
        
from hex.player import RandomPlayer
from collections import Counter
import random
from player import ModelPlayer
from display import animate_board_choices

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def run_only_reinforce(board_file, win_file):
    boards = []
    winners = []
    random_player1 = RandomPlayer()
    random_player2 = RandomPlayer()
    
    game_results = []

    for i in range(REINFORCEMENT_BATCH_SIZE):
        switch = bool(random.getrandbits(1))
        verbose = (i == 0)
        
        if switch:
            p_black = random_player1
            p_white = random_player2
        else:
            p_black = random_player2
            p_white = random_player1
        
        new_boards, new_winners, final_winner = play_game(p_black, p_white, verbose=verbose)
        
        if switch:
            if final_winner == 'black':
                game_results.append('rand1')
            else:
                game_results.append('rand2')
        else:
            if final_winner == 'black':
                game_results.append('rand1')
            else:
                game_results.append('rand2')
        
        boards.append(new_boards)
        winners.append(new_winners)
#         if verbose:
#             animate_board_choices('black-run-{}.mp4'.format(i), p_black.stats, BLACK)
#             animate_board_choices('white-run-{}.mp4'.format(i), p_white.stats, WHITE)
            
    
    print(Counter(game_results))
    np.save(board_file, boards)
    np.save(win_file, winners)
#     print(len(new_boards))
#     print(len(new_winners))
#     print(new_winners)

def apply_move(board, move, player, value):
    x, y = move
    player_idx = (BLACK - 1) if player == "black" else (WHITE - 1)
    
    board[x, y, player_idx] = value
    
    return board

def nice_pos_to_loc(pos_pair):
    return (pos_pair[0], ord(pos_pair[1].lower()) - 96)
    
def dec(t):
    return (t[0] - 1, t[1] - 1)
    
# run_only_reinforce()
file_nums = np.arange(0,REPEATS)
board_files = ["./RandData/boards"+str(num+1) for num in file_nums]
win_files = ["./RandData/winners"+str(num+1) for num in file_nums]
print(board_files)
print(win_files)

for num in file_nums:
    run_only_reinforce(board_files[num], win_files[num])