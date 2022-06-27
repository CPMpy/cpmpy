#!/usr/bin/python3
"""
Sudoku problem in CPMpy
"""

# load the libraries
import sys
sys.path.append("/Users/emiliogamba/Documents/01_VUB/01_Research/01_Shared_Projects/07_CPMPY")

import numpy as np
from cpmpy import *

e = 0 # value for empty cells
given = np.array([
    [6, 9, 4,  e, e, 1,  e, e, e],
    [e, e, 3,  e, 2, e,  e, 4, 5],
    [2, 7, e,  e, 6, e,  e, e, e],

    [e, e, 1,  e, e, 4,  e, 7, e],
    [e, 2, 6,  e, 8, 7,  e, 9, 3],
    [3, 5, 7,  e, e, e,  4, e, 2],

    [e, 6, 9,  e, e, e,  e, e, 1],
    [1, 3, e,  6, e, e,  7, e, e],
    [e, e, e,  1, e, 2,  e, e, e]])


# Variables
puzzle = intvar(1,9, shape=given.shape, name="puzzle")


model = Model(
    # Constraints on values (cells that are not empty)
    puzzle[given!=e] == given[given!=e], # numpy's indexing, vectorized equality
    # Constraints on rows and columns
    [AllDifferent(row) for row in puzzle],
    [AllDifferent(col) for col in puzzle.T], # numpy's Transpose
)

# Constraints on blocks
for i in range(0,9, 3):
    for j in range(0,9, 3):
        model += AllDifferent(puzzle[i:i+3, j:j+3]) # python's indexing


# Solve and print
if model.solve():
    #print(puzzle.value())
    # pretty print, mark givens with *
    out = ""
    for r in range(0,9):
        for c in range(0,9):
            out += str(puzzle[r,c].value())
            out += '* ' if given[r,c] else '  '
            if (c+1) % 3 == 0 and c != 8: # end of block
                out += '| '
        out += '\n'
        if (r+1) % 3 == 0 and r != 8: # end of block
            out += ('-'*9)+'+-'+('-'*9)+'+'+('-'*9)+'\n'
    print(out)
else:
    print("No solution found")
