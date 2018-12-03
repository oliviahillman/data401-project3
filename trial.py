#!/usr/bin/env python

from player import *
from models.basic import BasicHexModel

model = BasicHexModel()

BATCH_SIZE = 1000
NUM_EPOCHS = 10
STEPS_PER_EPOCH = 30

with open('data/board_states.npy', 'rb') as boards_file, \
    open('data/winners.npy', 'rb') as winners_file:
    boards = np.load(boards_file)
    winners = np.load(winners_file)
    
    winners_cat = np.array([[1.0, 0.0] if x==1 else [0.0, 1.0] for x in winners])

    print(boards[:100000, :, :, :].shape, winners_cat[:100000, :].shape)
    
    nums = np.arange(len(winners_cat))
    np.random.shuffle(nums)
    X_train =  boards[nums[:100000],:,:,:]
    y_train = winners_cat[nums[:100000],:]
    
    X_test =  boards[nums[100000:200000],:,:,:]
    y_test = winners_cat[nums[100000:200000],:]
    
    model.fit(X_train, y_train, verbose=1, batch_size = BATCH_SIZE,
              epochs=NUM_EPOCHS)
    
    game_results = []
    
    for i in range(20):
        if i == 0:
            new_boards, new_winners, final_winner = play_game(model, model, verbose=True)
        else:
            new_boards, new_winners, final_winner = play_game(model, model)
        
        game_results.append(final_winner)
    
    print(game_results)
    