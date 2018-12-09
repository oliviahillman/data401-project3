#!/usr/bin/env python

from hex.player import Player
from hex.game import Game

from convert_to_game_state import row_to_features

import random
import numpy as np

class ModelPlayer(Player):
    
    def __init__(self, hex_model):
        """Create a model based player, with a given model"""
        self.hex_model = hex_model

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
        
        converted_board = self.canonicalize_board(board)
        player = BLACK if self.role == "black" else WHITE
        if possibleMoves:
            possible_moves = [(pos, dec(nice_pos_to_loc(pos))) for pos in possibleMoves]
            boards = np.array([apply_move(converted_board, move, self.role) for _, move in possible_moves ])
            predictions = self.hex_model.predict(boards)
            
            if self.role == "black":
                chosen_move = possible_moves[np.argmax(predictions)][0]
            else:
                chosen_move = possible_moves[np.argmin(predictions)][0]
            
            self.stats['base'].append(converted_board)
            self.stats['preds'].append(flatten_predictions(converted_board, boards,
                                                           predictions, player - 1))
            self.stats['moves'].append(possible_moves)
            self.stats['count'] += 1

            return chosen_move
        else:
            raise Exception("No possible moves to make.")
            
    def canonicalize_board(self, board):
        given_board = board.getStateForPlayer(BLACK if self.role == "black" else WHITE)
        
        if self.role == "black":
            black_board = (given_board == 1).astype(np.int8)
            white_board = (given_board == -1).astype(np.int8)
        else:
            white_board = (given_board == 1).astype(np.int8)
            black_board = (given_board == -1).astype(np.int8)

        final_board = np.stack([black_board, white_board], axis=-1)
        
        return final_board
    
def flatten_predictions(base_board, new_boards, predictions, player):
    removed_common = new_boards - base_board[np.newaxis, :]
    scaled = removed_common * predictions[:, np.newaxis, np.newaxis]
    flattened = np.sum(scaled, axis=0)
    
    return flattened

def apply_move(board, move, player):
    board = board.copy()
    x, y = move
    player_idx = (BLACK - 1) if player == "black" else (WHITE - 1)
    
    board[x, y, player_idx] = 1
    
    return board

def nice_pos_to_loc(pos_pair):
    return (pos_pair[0], ord(pos_pair[1].lower()) - 96)
    
def dec(t):
    return (t[0] - 1, t[1] - 1)

BLACK = 1
WHITE = 2

def play_game(p1, p2, verbose=False):    
    game = Game(p1, p2)
    
    move_list, winner = game.play(verbose=verbose)
    game_states = row_to_features(move_list, winner, flipped=True)
    boards = np.array(game_states['boards'])
    
    stretched_winner = np.array([1 if game_states['winner']==BLACK else 0 \
                                 for i in range(len(game_states['boards']))])
    
    return boards, stretched_winner, winner
