import numpy as np

def convert_input(player, maze, opponent, mazeHeight, mazeWidth, piecesOfCheese):
    im_size = (2*mazeHeight-1,2*mazeWidth-1,1)
    canvas = np.zeros(im_size)
    (x,y) = player
    center_x, center_y = mazeWidth-1, mazeHeight-1
    for (x_cheese,y_cheese) in piecesOfCheese:
        canvas[y_cheese+center_y-y,x_cheese+center_x-x,0] = 1
    return canvas