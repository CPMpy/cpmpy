#!/usr/bin/python
"""
Sudoku problem in CPPY.

This is a straightforward implementation of Sudoku.

Based on the Numberjack model of Hakan Kjellerstrand

"""
from cppy import *
import numpy

# Problem data.
n = 9
puzzle = numpy.array([
    [0, 0, 0, 2, 0, 5, 0, 0, 0],
    [0, 9, 0, 0, 0, 0, 7, 3, 0],
    [0, 0, 2, 0, 0, 9, 0, 6, 0],
    [2, 0, 0, 0, 0, 0, 4, 0, 9],
    [0, 0, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 9, 0, 0, 0, 0, 0, 1],
    [0, 8, 0, 4, 0, 0, 1, 0, 0],
    [0, 6, 3, 0, 0, 0, 0, 8, 0],
    [0, 0, 0, 6, 0, 8, 0, 0, 0]])

# Construct the model.
x = IntVar(1, n, puzzle.shape)

c_val = [] # constraint on values
for index, v in np.ndenumerate(puzzle):
    if v != 0:
        c_val.append( x[index] != v )

# constraints on rows and columns
c_row = [alldifferent(row) for row in puzzle]
c_col = [alldifferent(col) for row in puzzle.T]

c_block = [] # constraint on blocks
reg = numpy.sqrt(n)
for i in xrange(0,n,reg):
    for j in xrange(0,n,reg):
        c_block.append( alldifferent(puzzle[i:i+3, j:j+3]) )

model = Model(c_val, c_row, c_col, c_block)

stats = model.solve()
print x.value
