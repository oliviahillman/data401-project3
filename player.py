#!/usr/bin/env python

from hex.player import Player
from hex.game import Game

from convert_to_game_state import row_to_features

import random
import numpy as np

class ModelPlayer(Player):
    
    def __init__(self, hex_model, relative=False):
        """Create a model based player, with a given model"""
        self.hex_model = hex_model
        self.relative = relative

        self.stats = {
            'base': [],
            'preds': [],
            'moves': [],
            'count': 0
        }
        
    def reset_stats(self):
        self.stats = {
            'base': [],
            'preds': [],
            'moves': [],
            'count': 0
        }
    
    def move(self, board):
        possibleMoves = board.getPossibleMoves()
        random.shuffle(possibleMoves)
        
        player = BLACK if self.role == "black" else WHITE
        given_board = board.getStateForPlayer(player)

        if not self.relative:
            converted_board = fixed_board_transform(given_board, player)
        else:
            converted_board = relative_board_transform(given_board, player)
        
        if possibleMoves:
            possible_moves = [(pos, dec(nice_pos_to_loc(pos))) for pos in possibleMoves]
            boards = np.array([apply_move(converted_board, move, self.role, self.relative) \
                               for _, move in possible_moves ])
            predictions = self.hex_model.predict(boards)

            if not self.relative:
                if self.role == "black":
                    chosen_move = possible_moves[np.argmax(predictions)][0]
                else:
                    chosen_move = possible_moves[np.argmin(predictions)][0]
            else:
                chosen_move = possible_moves[np.argmax(predictions)][0]
            
            self.stats['base'].append(converted_board)
            self.stats['preds'].append(flatten_predictions(converted_board,
                                                           boards,
                                                           predictions,
                                                           player - 1,
                                                           self.relative))
            self.stats['moves'].append(possible_moves)
            self.stats['count'] += 1

            return chosen_move
        else:
            raise Exception("No possible moves to make.")
            
def fixed_board_transform(given_board, player):
    if player == BLACK:
        black_board = (given_board == 1).astype(np.int8)
        white_board = (given_board == -1).astype(np.int8)
    else:
        white_board = (given_board == 1).astype(np.int8)
        black_board = (given_board == -1).astype(np.int8)
    
    final_board = np.stack([black_board, white_board], axis=-1)
    
    return final_board

def relative_board_transform(given_board, player):
    self_board = (given_board == 1).astype(np.int8)
    other_board = (given_board == -1).astype(np.int8)
    color_board = np.full_like(self_board, player - 1, dtype=np.int8)
    
    final_board = np.stack([self_board, other_board, color_board], axis=-1)
    
    return final_board
    
def flatten_predictions(base_board, new_boards, predictions, player, relative):
    removed_common = new_boards - base_board[np.newaxis, :]
    scaled = removed_common * predictions[:, np.newaxis, np.newaxis]
    flattened = np.sum(scaled, axis=0)

    return flattened

def apply_move(board, move, player, relative):
    board = board.copy()
    x, y = move
    player_idx = (BLACK - 1) if player == "black" else (WHITE - 1)
    
    if not relative:
        board[x, y, player_idx] = 1
    else:
        board[x, y, 0] = 1
    
    return board

def nice_pos_to_loc(pos_pair):
    return (pos_pair[0], ord(pos_pair[1].lower()) - 96)
    
def dec(t):
    return (t[0] - 1, t[1] - 1)

BLACK = 1
WHITE = 2

def fixed_to_relative_transform(board, player):
    black_board = board[:, :, 0]
    white_board = board[:, :, 1]
    color_board = np.full_like(black_board, player - 1)

    if player == BLACK:
        relative_board = np.stack([black_board, white_board, color_board], axis=-1)
    else:
        relative_board = np.stack([white_board, black_board, color_board], axis=-1)
    
    return relative_board

def convert_game_fixed_to_relative(boards, winner):
    players = np.empty((len(boards),))
    players[::2] = BLACK
    players[1::2] = WHITE
        
    relative_boards = np.array(
        [fixed_to_relative_transform(b, p) for b, p in zip(boards, players)]
    )

    stretched_winner = np.full(boards.shape[0], 1, dtype=np.int8)
    if winner == 'black':
        stretched_winner[1::2] = 0
    else:
        stretched_winner[0::2] = 0
        
    return relative_boards, stretched_winner

def play_game(p1, p2, verbose=False, relative=False):    
    game = Game(p1, p2)
    
    move_list, winner = game.play(verbose=verbose)
    game_states = row_to_features(move_list, winner, flipped=True)
    boards = np.array(game_states['boards'])
    
    if not relative:
        stretched_winner = np.array([1 if game_states['winner']==BLACK else 0 \
                                     for i in range(len(game_states['boards']))])
    else:
        boards, stretched_winner = convert_game_fixed_to_relative(boards, winner)
    
    return boards, stretched_winner, winner
