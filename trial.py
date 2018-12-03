#!/usr/bin/env python

from player import *
from models.basic import BasicHexModel

model = BasicHexModel()
print(play_game(model, model))