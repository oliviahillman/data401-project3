import pandas as pd
import numpy as np
import re
import tensorflow as tf

DATA_CSV = "/home/jovyan/data400_share/share/hex_scrapes/joineddata.csv"
OUTPUT_TF_RECORD = "data/full_dataset.tfrecord"

SPLIT_RE = re.compile(r"(\*|[a-zA-Z][0-9]+)")

BLACK = 1
WHITE = 2

def nice_pos_to_loc(pos_str):
    if pos_str != '*':
        alpha_val = ord(pos_str[0].lower()) - 96
        return (alpha_val, int(pos_str[1:]))
    else:
        return (pos_str, None)
    
def dec(t):
    return (t[0] - 1, t[1] - 1)

def swap(t):
    return (t[1], t[0])

def row_to_features(row):
    move_list = row[1][4]
    moves = list(map(nice_pos_to_loc, filter(lambda s: len(s) > 0, SPLIT_RE.split(move_list))))
    
    board = np.zeros((13, 13), dtype=np.int8)
    current_player = BLACK
    if len(moves) > 1 and moves[1][0] == '*':
        board[swap(dec(moves[0]))] = WHITE

        moves = moves[2:]
        current_player = BLACK
    else:
        board[dec(moves[0])] = BLACK

        moves = moves[1:]
        current_player = WHITE
        
    for move in moves:
        board[dec(move)] = current_player
        
        if current_player == WHITE:
            current_player = BLACK
        else:
            current_player = WHITE
    
    winner = WHITE if row[1][5] == 'white' else BLACK
    
    winner_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes([winner])]))
    board_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(board.flatten())]))
    
    return {
        'winner': winner_feature,
        'board': board_feature
    }

def main():
    df = pd.read_csv(DATA_CSV).astype({'gid': np.int64}).set_index('gid')

    with tf.python_io.TFRecordWriter(OUTPUT_TF_RECORD) as writer:
        for row in df.iterrows():
            features = tf.train.Features(feature=row_to_features(row))
            example = tf.train.Example(features=features)

            writer.write(example.SerializeToString())

if __name__ == '__main__':
    main()