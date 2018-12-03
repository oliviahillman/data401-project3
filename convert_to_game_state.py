#!/usr/bin/env python

import pandas as pd
import numpy as np
import re
import tensorflow as tf

DATA_CSV = "/home/jovyan/data400_share/share/hex_scrapes/joineddata.csv"
OUTPUT_TF_RECORD = "data/full_dataset.tfrecord"

SPLIT_RE = re.compile(r"(\*|[a-zA-Z][0-9]+)")

SPLIT_FLIP_RE = re.compile(r"(\*|[0-9]+[a-zA-Z])")

BLACK = 0
WHITE = 1

ONCE = True

def nice_pos_to_loc(pos_str, flipped=False):
    if pos_str != '*':
        if flipped:
            alpha_val = ord(pos_str[-1].lower()) - 96
            num_val = int(pos_str[:-1])
        else:
            alpha_val = ord(pos_str[0].lower()) - 96
            num_val = int(pos_str[1:])
        return (alpha_val, num_val)
    else:
        return (pos_str, None)
    
def dec(t):
    return (t[0] - 1, t[1] - 1)

def swap(t):
    return (t[1], t[0])

def row_to_features(move_list, winner_str, flipped=False):
    reg = SPLIT_RE if not flipped else SPLIT_FLIP_RE
    
    filtered_moves = filter(lambda s: len(s) > 0, reg.split(move_list))
    
    if flipped:
        moves = list(map(lambda pos: nice_pos_to_loc(pos, flipped=True), filtered_moves))
    else:
        moves = list(map(nice_pos_to_loc, filtered_moves))
    
    boards = []
    
    board = np.zeros((13, 13, 2), dtype=np.int8)
    current_player = BLACK
    if len(moves) > 1 and moves[1][0] == '*':
        x, y = swap(dec(moves[0]))
        board[x, y, WHITE] = 1

        moves = moves[2:]
        current_player = BLACK
    else:
        x, y = dec(moves[0])
        board[x, y, BLACK] = 1

        moves = moves[1:]
        current_player = WHITE
        
    boards.append(board)
        
    for move in moves:
        board = board.copy()
        x, y = dec(move)
        board[x, y, current_player] = 1
        
        boards.append(board)
        
        if current_player == WHITE:
            current_player = BLACK
        else:
            current_player = WHITE
    
    winner = 1 if winner_str == 'black' else 0
    
    return {
        'boards': boards,
        'winner': winner
    }

def main():
    df = pd.read_csv(DATA_CSV).astype({'gid': np.int64}).set_index('gid')
    
    board_states = []
    winners = []
    
    for row in df.iterrows():
        move_list = row[1][4]
        winner_str = row[1][5]
        features = row_to_features(move_list, winner_str)

        board_states.extend(features['boards'])
        winners.extend([features['winner']] * len(features['boards']))
        
    board_states = np.array(board_states)
    winners = np.array(winners).reshape((len(winners), 1))
    
    print(board_states.shape, winners.shape)
    
    with open('data/board_states.npy', 'wb') as boards_file, \
        open('data/winners.npy', 'wb') as winners_file:
        np.save(boards_file, board_states)
        np.save(winners_file, winners)

if __name__ == '__main__':
    main()