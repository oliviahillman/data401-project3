#!/usr/bin/env python

from player import *

NUM_EPOCHS = 3
REINFORCEMENT_ROUNDS = 12
REINFORCEMENT_BATCH_SIZE = 256 + 1
RELATIVE = True
TRAIN_REINFORCE = True
TRAIN_INITIAL = True

GAME_LEN_INITIAL_LOWER_CUTOFF = 0
GAME_LEN_INITIAL_UPPER_CUTOFF = 127

GAME_LEN_REINFORCE_LOWER_CUTOFF = 0
GAME_LEN_REINFORCE_UPPER_CUTOFF = 136

TARGET_NUM_TRAIN_EXAMPLES = 25000
        
from hex.player import RandomPlayer
from collections import Counter
import random
from models.declan import DeclanModel
from player import ModelPlayer, convert_game_fixed_to_relative
from display import animate_board_choices
import tensorflow as tf

from tensorflow.keras import backend as K
K.set_session(
    tf.Session(
        config=tf.ConfigProto(intra_op_parallelism_threads=32,
                                inter_op_parallelism_threads=32)))


from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.train import checkpoint_exists

import os
from glob import glob

os.environ['KMP_DUPLICATE_LIB_OK']='True'

RAND_DATA_FOLDER = 'RandData/'
BOARD_FILE_FORMAT = RAND_DATA_FOLDER + 'boards{}.npy'
WINNER_FILE_FORMAT = RAND_DATA_FOLDER + 'winners{}.npy'

BLACK = 1
WHITE = 2

def get_acceptable_training_data(relative=False):
    accepted_games = []
    accepted_winners = []
    for ind in range(1, 60 + 1):
        board_filename = BOARD_FILE_FORMAT.format(ind)
        winner_filename = WINNER_FILE_FORMAT.format(ind)
        
        raw_boards = np.load(board_filename)
        raw_winners = np.load(winner_filename)
        
        for g in range(len(raw_boards)):
            move_count = len(raw_boards[g])
            
            if GAME_LEN_INITIAL_LOWER_CUTOFF < move_count < GAME_LEN_INITIAL_UPPER_CUTOFF:
                if not relative:
                    accepted_games.append(raw_boards[g])
                    accepted_winners.append(raw_winners[g])
                else:
                    winner = "black" if (raw_winners[g] == (BLACK - 1)).all() else "white"
                    boards, winners = convert_game_fixed_to_relative(raw_boards[g], winner)
                    accepted_games.append(boards)
                    accepted_winners.append(winners)
    
    print("Accepted {} games".format(len(accepted_games)))
    accepted_games = np.concatenate(accepted_games)
    accepted_winners = np.concatenate(accepted_winners)
    print(accepted_games.shape, accepted_winners.shape)
    
    return accepted_games, accepted_winners

def run():
    callbacks = [
        ModelCheckpoint('checkpoints/declan_model_weights', monitor='val_loss', verbose=0,
                        save_best_only=True, save_weights_only=True, mode='auto', period=1)
    ]
    model = DeclanModel()
    initial_boards, initial_winners = get_acceptable_training_data(relative=RELATIVE)

    if checkpoint_exists('checkpoints/declan_model_weights'):
        print("Loading from checkpoint")
        model.load('checkpoints/declan_model_weights')
    elif TRAIN_INITIAL:
        model.fit(initial_boards, initial_winners,
                  epochs=NUM_EPOCHS, validation_split=0.1,
                  callbacks=callbacks, shuffle=True)
    
    p1 = ModelPlayer(model, relative=RELATIVE)
    p2 = ModelPlayer(model, relative=RELATIVE)
    
    for r in range(REINFORCEMENT_ROUNDS):
        accumulated_boards = None
        accumulated_winners = None
        win_results_by_color = Counter()
        win_results_by_player = Counter()
        accept_counter = 0

        for i in range(REINFORCEMENT_BATCH_SIZE):
            switch = bool(random.getrandbits(1))
            verbose = (i % 64 == 0)

            if switch:
                p_black = p1
                p_white = p2
            else:
                p_black = p2
                p_white = p1

            new_boards, new_winners, final_winner = play_game(p_black, p_white,
                                                              verbose=False,
                                                              relative=RELATIVE)
            win_results_by_color[final_winner] += 1
            accept_game = GAME_LEN_REINFORCE_LOWER_CUTOFF < len(new_boards) < GAME_LEN_REINFORCE_UPPER_CUTOFF
            print("{}: {}".format(i, len(new_boards)), end=', ', flush=True)
            
            if accept_game:
                accept_counter += 1
                if accumulated_boards is None:
                    accumulated_boards = new_boards
                else:
                    accumulated_boards = np.concatenate((accumulated_boards, new_boards))

                if accumulated_winners is None:
                    accumulated_winners = new_winners
                else:
                    accumulated_winners = np.concatenate((accumulated_winners, new_winners))

            if verbose:
                print("\nOutputing {} {} animation".format(r, i))
                print("This game was accepted: {}".format(accept_game))
                animate_board_choices('outputs/black-{}-{}.mp4'.format(r, i),
                                      p_black.stats, BLACK,
                                      relative=RELATIVE)
                animate_board_choices('outputs/white-{}-{}.mp4'.format(r, i),
                                      p_white.stats, WHITE,
                                      relative=RELATIVE)
                
            if switch:
                if final_winner == 'black':
                    win_results_by_player['p1'] += 1
                else:
                    win_results_by_player['p2'] += 1
            else:
                if final_winner == 'black':
                    win_results_by_player['p2'] += 1
                else:
                    win_results_by_player['p1'] += 1
                
            p1.reset_stats()
            p2.reset_stats()
        
        print("Accepted {} out of {} games".format(accept_counter, REINFORCEMENT_BATCH_SIZE))
        print(win_results_by_color)
        print(win_results_by_player)
        print("Finished round {}".format(r))
        if TRAIN_REINFORCE:
            num_gathered = len(accumulated_boards)
            if num_gathered < TARGET_NUM_TRAIN_EXAMPLES:
                num_missing = TARGET_NUM_TRAIN_EXAMPLES - num_gathered
                extra_sample_indices = np.random.choice(len(initial_boards), size=num_missing)
                extra_boards = initial_boards[extra_sample_indices]
                extra_winners = initial_winners[extra_sample_indices]
                
                accumulated_boards = np.concatenate([accumulated_boards, extra_boards])
                accumulated_winners = np.concatenate([accumulated_winners, extra_winners])
            
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

if __name__ == '__main__':
    run()