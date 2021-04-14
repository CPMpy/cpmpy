#!/usr/bin/python3
"""
N-queens problem in CPMpy

CSPlib prob054

Problem description from the numberjack example:
The N-Queens problem is the problem of placing N queens on an N x N chess
board such that no two queens are attacking each other. A queen is attacking
another if it they are on the same row, same column, or same diagonal.
"""

# load the libraries
import numpy as np
from cpmpy import *

N = 8

# Variables (one per row)
queens = IntVar(1,N, shape=N)

# Constraints on columns and left/right diagonal
m = Model([
        alldifferent(queens),
        alldifferent([queens[i] - i for i in range(N)]),
        alldifferent([queens[i] + i for i in range(N)]),
    ])

if m.solve():
    # pretty print
    line = '+---'*N+'+\n'
    out = line
    for queen in queens.value():
        out += '|   '*(queen-1)+'| Q '+'|   '*(N-queen)+'|\n'
        out += line
    print(out)
else:
    print("No solution found")
