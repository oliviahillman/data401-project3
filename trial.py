#!/usr/bin/env python

from player import *
from models.basic import BasicHexModel

model = BasicHexModel()

BATCH_SIZE = 1000
NUM_EPOCHS = 10
STEPS_PER_EPOCH = 30

REINFORCEMENT_BATCH_SIZE = 32

SUBSET_SIZE = 10000

def run_start_w_training():
    with open('data/board_states.npy', 'rb') as boards_file, \
        open('data/winners.npy', 'rb') as winners_file:
        boards = np.load(boards_file)
        winners = np.load(winners_file)

        print(boards.shape, winners.shape)

        print(boards[:SUBSET_SIZE, :, :, :].shape, winners[:SUBSET_SIZE, :].shape)

        nums = np.arange(len(winners))
        np.random.shuffle(nums)
        X_train =  boards[nums[:SUBSET_SIZE],:,:,:]
        y_train = winners[nums[:SUBSET_SIZE],:]

        X_test =  boards[nums[SUBSET_SIZE:(2 * SUBSET_SIZE)],:,:,:]
        y_test = winners[nums[SUBSET_SIZE:(2 * SUBSET_SIZE)],:]

        model.fit(X_train, y_train, verbose=1, batch_size = BATCH_SIZE,
                  epochs=NUM_EPOCHS)

        game_results = []

        for i in range(REINFORCEMENT_BATCH_SIZE):
            new_boards, new_winners, final_winner = play_game(model, model, verbose=(i == 0))

            game_results.append(final_winner)

        print(game_results)

def run_only_reinforce():
    p_black = ModelPlayer(model)
    p_white = ModelPlayer(model)
    
    game_results = []

    for i in range(REINFORCEMENT_BATCH_SIZE):
        
        new_boards, new_winners, final_winner = play_game(p_black, p_white, verbose=(i == 0))
        game_results.append(final_winner)
        
        p_black.reset_stats()
        p_white.reset_stats()

    print(game_results)
    
    
run_only_reinforce()