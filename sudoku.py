#!/usr/bin/python3
"""
Sudoku problem in CPpy

Based on the Numberjack model of Hakan Kjellerstrand
"""
from cppy import *
import numpy

x = 0 # cells whose value we seek
n = 9 # matrix size
given = numpy.array([
    [x, x, x,  2, x, 5,  x, x, x],
    [x, 9, x,  x, x, x,  7, 3, x], 
    [x, x, 2,  x, x, 9,  x, 6, x],
        
    [2, x, x,  x, x, x,  4, x, 9],
    [x, x, x,  x, 7, x,  x, x, x],
    [6, x, 9,  x, x, x,  x, x, 1],
        
    [x, 8, x,  4, x, x,  1, x, x],
    [x, 6, 3,  x, x, x,  x, 8, x],
    [x, x, x,  6, x, 8,  x, x, x]])


# Variables
puzzle = IntVar(1, n, shape=given.shape)

constraints = []
# constraints on rows and columns
constraints += [ alldifferent(row) for row in puzzle ]
constraints += [ alldifferent(col) for col in puzzle.T ]

# constraint on blocks
for i in range(0,n,3):
    for j in range(0,n,3):
        constraints += [ alldifferent(puzzle[i:i+3, j:j+3]) ]

# constraints on values
constraints += [ puzzle[given!=x] == given[given!=x] ]

model = Model(constraints)
stats = model.solve()
print(stats)
print(puzzle.value())
