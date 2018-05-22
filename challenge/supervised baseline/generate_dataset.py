import os
import numpy as np
import tqdm
import ast
import utils
import scipy
import scipy.sparse

PHRASES = {
    "# Random seed\n": "seed",
    "# MazeMap\n": "maze",
    "# Pieces of cheese\n": "pieces"    ,
    "# Rat initial location\n": "rat"    ,
    "# Python initial location\n": "python"   , 
    "rat_location then python_location then pieces_of_cheese then rat_decision then python_decision\n": "play"
}

MOVE_DOWN = 'D'
MOVE_LEFT = 'L'
MOVE_RIGHT = 'R'
MOVE_UP = 'U'

translate_action = {
    MOVE_LEFT:0,
    MOVE_RIGHT:1,
    MOVE_UP:2,
    MOVE_DOWN:3
}

def process_file(filename):
    f = open(filename,"r")    
    info = f.readline()
    params = dict(play=list())
    while info is not None:
        if info.startswith("{"):
            params["end"] = ast.literal_eval(info)
            break
        if "turn " in info:
            info = info[info.find('rat_location'):]
        if info in PHRASES.keys():
            param = PHRASES[info]
            if param == "play":
                rat = ast.literal_eval(f.readline())
                python = ast.literal_eval(f.readline())
                pieces = ast.literal_eval(f.readline())
                rat_decision = f.readline().replace("\n","")
                python_decision = f.readline().replace("\n","")
                play_dict = dict(
                    rat=rat,python=python,piecesOfCheese=pieces,
                    rat_decision=rat_decision,python_decision=python_decision)
                params[param].append(play_dict)
            else:
                params[param] = ast.literal_eval(f.readline())
        else:
            print("did not understand:", info)
            break
        info = f.readline()
    return params

def dict_to_x_y(end,rat, python, maze, piecesOfCheese,rat_decision,python_decision,
                mazeWidth=21, mazeHeight=15):
    # We only use the winner
    if end["win_python"] == 1: 
        player = python
        opponent = rat        
        decision = python_decision
    elif end["win_rat"] == 1:
        player = rat
        opponent = python        
        decision = rat_decision
    else:
        return False
    if decision == "None" or decision == "": #No play
        return False
    x_1 = utils.convert_input(player, maze, opponent, mazeHeight, mazeWidth, piecesOfCheese)
    y = np.zeros((1,4),dtype=np.int8)
    y[0][translate_action[decision]] = 1
    return x_1,y


games = list()
directory = "saves/"
for root, dirs, files in os.walk(directory):
    for filename in tqdm.tqdm(files):
        if filename.startswith("."):
            continue
        game_params = process_file(directory+filename)
        games.append(game_params)

x_1_train = list()
y_train = list()
wins_python = 0
wins_rat = 0
for game in tqdm.tqdm(games):
    if game["end"]["win_python"] == 1: 
        wins_python += 1
    elif game["end"]["win_rat"] == 1:
        wins_rat += 1
    else:
        continue
    plays = game["play"]
    for play in plays:
        x_y = dict_to_x_y(**play,maze=game_params["maze"],end=game["end"])
        if x_y:
            x1, y = x_y
            y_train.append(scipy.sparse.csr_matrix(y.reshape(1,-1)))
            x_1_train.append(scipy.sparse.csr_matrix(x1.reshape(1,-1)))
print("Greedy/Draw/Random Greedy, {}/{}/{}".format(wins_rat,1000 - wins_python - wins_rat, wins_python)) 

import pickle
pickle.dump([x_1_train,y_train],open("pyrat_dataset.pkl","wb"))