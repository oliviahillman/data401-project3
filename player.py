#!/usr/bin/env python

from hex.player import Player
from hex.game import Game

import random
import numpy as np

class ModelPlayer(Player):
    
    def __init__(self, ident, predictor):
        """Create a model based player, with a given identity.
        
        ident is either 1 (BLACK) or 2 (WHITE)
        """
        self.ident = ident
        self.board = np.zeros((13, 13), dtype=np.int8)
        self.predictor = predictor
    
    def move(self, board):
        possibleMoves = board.getPossibleMoves()
        if possibleMoves:
            converted_moves = ((pos, dec(nice_pos_to_loc(pos))) for pos in possibleMoves)
            
            new_board_positions = [(nice_pos, *self.eval_move(move)) for nice_pos, move in converted_moves]
            
            chosen_move, value, new_board = max(new_board_positions,
                                                key=lambda new_pos: new_pos[1][self.ident - 1])
            
            self.board = new_board

            return chosen_move
        else:
            raise Exception("No possible moves to make.")
            
    def eval_move(self, move):
        board = self.board.copy()
        
        board[move] = self.ident
        
        result = self.predictor(board)
        
        return (result, board)
            
def nice_pos_to_loc(pos_pair):
    return (pos_pair[0], ord(pos_pair[1].lower()) - 96)
    
def dec(t):
    return (t[0] - 1, t[1] - 1)

BLACK = 1
WHITE = 2

def main():
    model = lambda board: random.choice([[0, 1], [1, 0]])
    p1 = ModelPlayer(BLACK, model)
    p2 = ModelPlayer(WHITE, model)
    
    game = Game(p1, p2)
    
    game.play(verbose=True)

if __name__ == '__main__':
    main()