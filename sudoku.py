#!/usr/bin/python3
"""
Sudoku problem in CPpy

Based on the Numberjack model of Hakan Kjellerstrand
"""
from cppy import *
import numpy

x = 0 # cells whose value we seek
puzzle = numpy.array([
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
(n,_) = puzzle.shape # get matrix dimensions
x = IntVar(1, n, puzzle.shape)


# constraints on values
constr_values = ( x[puzzle>0] == puzzle[puzzle>0] )

# constraints on rows and columns
constr_row = [alldifferent(row) for row in x]
constr_col = [alldifferent(col) for col in x.T]

# constraint on blocks
constr_block = [] 
for i in range(0,n,3):
    for j in range(0,n,3):
        constr_block.append( alldifferent(x[i:i+3, j:j+3]) )


model = Model(constr_values, constr_row, constr_col, constr_block)
stats = model.solve()

print(model)
print(stats)
