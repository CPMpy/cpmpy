#!/usr/bin/python3
"""
Minesweeper problem in CpMPy

Based on the Minesweeper model of Hakan Kjellerstrand
http://www.hakank.org/common_cp_models/#minesweeper

A specification is a square matrix of characters. Alphanumeric 
characters represent the number of mines adjacent to that field. 
X represent fields with an unknown number of mines adjacent to 
it (or an actual mine).
'''

E.g.
     "XX2X3X"
     "2XXXXX"
     "XX24X3"
     "1X34XX"
     "XXXXX3"
     "X3X3XX"

"""

from cpmpy import *
import numpy

X = -1
default_game =  numpy.array([
            [2,3,X,2,2,X,2,1],
            [X,X,4,X,X,4,X,2],
            [X,X,X,X,X,X,4,X],
            [X,5,X,6,X,X,X,2],
            [2,X,X,X,5,5,X,2],
            [1,3,4,X,X,X,4,X],
            [0,1,X,4,X,X,X,3],
            [0,1,2,X,2,3,X,2]
            ])

r_ , c_ = 8,8
S = [-1,0,1] # for the neighbors of a cell
# Variables
mines = IntVar(0, 1, shape=default_game.shape) 

constraint = []

# constraint += [ mines[default_game>X] == 0 ]
for i in range(r_):
    for j in range(c_):
        if default_game[i,j] >=0:
            constraint += [ np.sum( [mines[i+a,j+b] for a in S for b in S
            if i+a >=0 and i+a <r_ and j+b >=0 and j+b<c_])==default_game[i][j] ]
        # if default_game[i,j] >=0:
            constraint += [ mines[i,j] == 0 ] # This cell cannot be a mine


model = Model(constraint)
stats = model.solve()
print(stats)
print(mines.value())