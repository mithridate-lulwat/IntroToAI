# Template file to create an AI for the game PyRat
# http://formations.telecom-bretagne.eu/pyrat

###############################
# Team name to be displayed in the game 
TEAM_NAME = "KerasAI"

###############################
# When the player is performing a move, it actually sends a character to the main program
# The four possibilities are defined here
MOVE_DOWN = 'D'
MOVE_LEFT = 'L'
MOVE_RIGHT = 'R'
MOVE_UP = 'U'

###############################
# Please put your imports here

import numpy as np
import random as rd
import pickle
import time
from utils import convert_input
from sklearn.externals import joblib


###############################
# Please put your global variables here

# Global variables
global model
    
###############################
# Preprocessing function
# The preprocessing function is called at the start of a game
# It can be used to perform intensive computations that can be
# used later to move the player in the maze.
###############################
# Arguments are:
# mazeMap : dict(pair(int, int), dict(pair(int, int), int))
# mazeWidth : int
# mazeHeight : int
# playerLocation : pair(int, int)
# opponentLocation : pair(int,int)
# piecesOfCheese : list(pair(int, int))
# timeAllowed : float
###############################
# This function is not expected to return anything
def preprocessing(mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, piecesOfCheese, timeAllowed):
    global model
    model = joblib.load('save.pkl') 

###############################
# Turn function
# The turn function is called each time the game is waiting
# for the player to make a decision (a move).
###############################
# Arguments are:
# mazeMap : dict(pair(int, int), dict(pair(int, int), int))
# mazeWidth : int
# mazeHeight : int
# playerLocation : pair(int, int)
# opponentLocation : pair(int, int)
# playerScore : float
# opponentScore : float
# piecesOfCheese : list(pair(int, int))
# timeAllowed : float
###############################
# This function is expected to return a move
def turn(mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese, timeAllowed):    
    global model,input_tm1, action, score
    input_t = convert_input(playerLocation, mazeMap, opponentLocation, mazeHeight, mazeWidth, piecesOfCheese)    
    output = model.predict(input_t.reshape(1,-1))
    action = output[0]
    return [MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN][action]

def postprocessing (mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese, timeAllowed):
    pass    
