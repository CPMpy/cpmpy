#!/usr/bin/python3
"""
Sudoku problem in CPMpy
"""

# load the libraries
import numpy as np
from cpmpy import *

e = 0 # value for empty cells
given = np.array([
    [e, e, e,  2, e, 5,  e, e, e],
    [e, 9, e,  e, e, e,  7, 3, e],
    [e, e, 2,  e, e, 9,  e, 6, e],

    [2, e, e,  e, e, e,  4, e, 9],
    [e, e, e,  e, 7, e,  e, e, e],
    [6, e, 9,  e, e, e,  e, e, 1],

    [e, 8, e,  4, e, e,  1, e, e],
    [e, 6, 3,  e, e, e,  e, 8, e],
    [e, e, e,  6, e, 8,  e, e, e]])


# Variables
puzzle = IntVar(1,9, shape=given.shape)

constraints = []
# Constraints on rows and columns
constraints += [ alldifferent(row) for row in puzzle ]
constraints += [ alldifferent(col) for col in puzzle.T ] # numpy's Transpose

# Constraints on blocks
for i in range(0,9, 3):
    for j in range(0,9, 3):
        constraints += [ alldifferent(puzzle[i:i+3, j:j+3]) ] # python's indexing

# Constraints on values (cells that are not empty)
constraints += [ puzzle[given!=e] == given[given!=e] ] # numpy's indexing


# Solve and print
model = Model(constraints)
model.solve()

print(puzzle.value())
