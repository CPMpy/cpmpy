#!/usr/bin/python3
"""
N-puzzle problem in CPMpy

Reworked based on Hakan Kjellerstrand's
https://github.com/hakank/hakank/blob/master/minizinc/n_puzzle.mzn

A typical children toy, where a picture is divided in n+1 blocks and mangled, and the goal is to trace the steps to the original picture.
"""

# load the libraries
import math
import numpy as np
from cpmpy import *

# Data
# '0' is empty spot
puzzle_start = np.array([
    [0,3,6],
    [2,4,8],
    [1,7,5]]).reshape(-1)
puzzle_end = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,0]]).reshape(-1)
t = puzzle_start.shape[0] # flat length

# max nr steps to the solution
num_sols = 20

# valid moves, INCLUDING staying at the same position
valid_moves_sm = cparray([  # would be better of with table?
   [1,1,0,1,0,0,0,0,0], # 0
   [1,1,1,0,1,0,0,0,0], # 1
   [0,1,1,0,0,1,0,0,0], # 2
   [1,0,0,1,1,0,1,0,0], # 3
   [0,1,0,1,1,1,0,1,0], # 4
   [0,0,1,0,1,1,0,0,1], # 5
   [0,0,0,1,0,0,1,1,0], # 6
   [0,0,0,0,1,0,1,1,1], # 7
   [0,0,0,0,0,1,0,1,1], # 8
])

# Variables
x = IntVar(0,t-1, shape=(num_sols,t))

# the moves, index in puzzle
move_from = IntVar(0,t-1, shape=(num_sols)) # is move_to[i-1]... could remove
move_to = IntVar(0,t-1, shape=(num_sols))

# is this row the solution?
check = BoolVar(shape=num_sols)

# index of first solution
check_ix = IntVar(0,num_sols)


def same(x, y):
    return all(x == y)

m = Model(minimize=check_ix)

# start and end puzzle
m += [
    same(x[0, :], puzzle_start),
    same(x[num_sols-1, :], puzzle_end),
]

# in each puzzle, all cells different
m += [ alldifferent(x[i,:]) for i in range(num_sols) ]


# move_to is location of the empty piece
m += [ x[i, move_to[i]] == 0 for i in range(num_sols) ]

# move_from is previous move_to, except first
m += [ move_from[0] == move_to[0] ]
m += [ move_from[i] == move_to[i-1] for i in range(1,num_sols) ]

# only move_from/move_to can have a change in x
m += [
    ((move_from[i] != j) & (move_to[i] != j)).implies(x[i,j] == x[i-1,j])
for i in range(1,num_sols) for j in range(t) ]


# require valid moves (including to same position)
m += [ valid_moves_sm[ (t*move_from[i]+move_to[i]) ] == 1 for i in range(num_sols)]


# check whether this is the end solution
m += [ check[i] == same(x[i,:], puzzle_end) for i in range(num_sols)]

# once the solution is found, keep it
m += [ check[i] >= check[i-1] for i in range(1,num_sols) ]

# index of a solution (when minimized, index of first solution)
m += [ check[check_ix] == 1 ]


# visualisation helper function
def visu_npuzzle(xval, pos_from, pos_to):
    dim = int(math.sqrt(t))
    assert (dim*dim == t), "non-square matrix??"
    out = ""
    for r in range(dim):
        for c in range(dim):
            pos = r*dim+c
            if pos == pos_from or pos == pos_to:
                out += f"{xval[pos]}* "
            else:
                out += f"{xval[pos]}  "
        out += '\n'
    print(out)

if not m.solve() is False:
    for i in range(check_ix.value()+1):
        visu_npuzzle(x[i,:].value(), move_from[i].value(), move_to[i].value())
    print(f"Found in {check_ix.value()} steps")
else:
    print("UNSAT, try increasing nr of steps? or wrong input...")
print(m.status())
