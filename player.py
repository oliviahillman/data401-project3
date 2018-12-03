#!/usr/bin/env python

from hex.player import Player
from hex.game import Game

from convert_to_game_state import row_to_features

import random
import numpy as np

class ModelPlayer(Player):
    
    def __init__(self, ident, hex_model):
        """Create a model based player, with a given identity.
        
        ident is either 1 (BLACK) or 2 (WHITE)
        """
        self.ident = ident
        self.board = np.zeros((13, 13, 1), dtype=np.int8)
        self.hex_model = hex_model
    
    def move(self, board):
        possibleMoves = board.getPossibleMoves()
        random.shuffle(possibleMoves)
        if possibleMoves:
            possible_moves = [(pos, dec(nice_pos_to_loc(pos))) for pos in possibleMoves]
            boards = np.array([self.apply_move(move) for _, move in possible_moves ])
            predictions = self.hex_model.predict(boards)
            
            chosen_idx = np.argmax(predictions[:, self.ident - 1])
            
            new_board = boards[chosen_idx]
            chosen_move = possible_moves[chosen_idx][0]
            
            self.board = new_board

            return chosen_move
        else:
            raise Exception("No possible moves to make.")
            
    def apply_move(self, move):
        board = self.board.copy()
        
        board[move] = self.ident
        
        return board

def nice_pos_to_loc(pos_pair):
    return (pos_pair[0], ord(pos_pair[1].lower()) - 96)
    
def dec(t):
    return (t[0] - 1, t[1] - 1)

BLACK = 1
WHITE = 2

def play_game(model_black, model_white, verbose=False):
    p1 = ModelPlayer(BLACK, model_black)
    p2 = ModelPlayer(WHITE, model_white)
    
    game = Game(p1, p2)
    
    move_list, winner = game.play(verbose=verbose)
    game_states = row_to_features(move_list, winner, flipped=True)
    boards = np.array(game_states['boards'])
    
    stretched_winner = np.array([[1.0, 0.0] if game_states['winner']==BLACK else [0.0, 1.0] \
                                 for i in range(len(game_states['boards']))])
    
    return boards, stretched_winner, winner
