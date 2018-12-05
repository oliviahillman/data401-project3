#!/usr/bin/env python

import pandas as pd
import numpy as np
import re

games_df = pd.read_csv('data/games.csv')
games_df.head()

games_df['black'] = games_df['black'].str.replace('★', '').str.strip()
games_df['white'] = games_df['white'].str.replace('★', '').str.strip()

SPLIT_RE = re.compile(r"(\*|[a-zA-Z][0-9]+)")

def count_moves(move_list):
    filtered_moves = filter(lambda s: len(s) > 0, SPLIT_RE.split(move_list))
    
    return len(list(filtered_moves))

games_df['num_moves'] = games_df['move_list'].apply(count_moves)
reduced_df = games_df[games_df['num_moves'] > 3]

reduced_df.to_csv('data/reduced_games.csv', index=False)

