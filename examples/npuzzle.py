#!/usr/bin/python3
"""
N-puzzle problem in CPMpy

Reworked based on Hakan Kjellerstrand's
https://github.com/hakank/hakank/blob/master/minizinc/n_puzzle.mzn

A typical children toy, where a picture is divided in n+1 blocks and mangled, and the goal is to trace the steps to the original picture.
"""

# load the libraries
import numpy as np
from cpmpy import *

# Data
t = 9 # flat puzzle length
# '0' is empty spot
puzzle_start = np.array([
    [0,3,6],
    [2,4,8],
    [1,7,5]]).reshape(-1)
puzzle_end = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,0]]).reshape(-1)

# max nr steps to the solution
num_sols = 20

valid_moves = cparray([
   [0,1,0,1,0,0,0,0,0], # 0
   [1,0,1,0,1,0,0,0,0], # 1
   [0,1,0,0,0,1,0,0,0], # 2
   [1,0,0,0,1,0,1,0,0], # 3
   [0,1,0,1,0,1,0,1,0], # 4
   [0,0,1,0,1,0,0,0,1], # 5
   [0,0,0,1,0,0,0,1,0], # 6
   [0,0,0,0,1,0,1,0,1], # 7
   [0,0,0,0,0,1,0,1,0], # 8
])

# Variables
x = IntVar(0,t-1, shape=(num_sols,t))

# the moves, index in puzzle
move_from = IntVar(0,t-1, shape=(num_sols))
move_to = IntVar(0,t-1, shape=(num_sols))

# is this row the solution?
check = BoolVar(shape=num_sols)

# index of first solution
check_ix = IntVar(0,num_sols)

# auxiliary: the nr of changes
changes = IntVar(0,2, shape=num_sols)


def same(x, y):
    return all(x == y)

m = Model(minimize=check_ix)


# initialisations
m += [
    same(x[0, :], puzzle_start),
    x[0, move_from[0]] == 0, # location of empty block
    move_to[0] == move_from[0],
    ~check[0],
    changes[0] == 0,
]

# last row should be end puzzle (this is the goal)
m += [ same(x[num_sols-1, :], puzzle_end) ]

# select the operations
for i in range(1, num_sols):
    m += [ alldifferent(x[i,:]) ]
    
    # at most 2 changes of each move
    m += [
        changes[i] == sum(x[i,:] != x[i-1,:]),
        ((changes[i] == 2) | (changes[i] == 0)),
    ]

    # is this row a solution Then we're done
    m += [ check[i] == same(x[i,:], puzzle_end) ]

    # select operations
    # either the row is same as last line
    same_as_last = all([
        same(x[i-1,:], x[i,:]),
        move_to[i-1] == move_from[i],
        move_to[i] == move_from[i],
    ])
    # or we move a piece
    move_a_piece = all([
        move_to[i-1] == move_from[i],
        #valid_moves[moves[i,1], moves[i,2]], # two indices not allowed
        valid_moves[ (t*move_from[i]+move_to[i]) ],
        x[i, move_to[i]] == 0, # location of empty block
        move_from[i] != move_to[i],
    ])
    m += [same_as_last | move_a_piece]

# once found, stay the same; get (first) found index
m += [ check[i] >= check[i-1] for i in range(1,num_sols) ]
m += [ check[check_ix] == 1 ]

if not m.solve() is False:
    print(f"Found in {check_ix.value()} steps")
    # TODO: highlight changes?
    for i in range(check_ix.value()+1):
        print(x[i,:].value().reshape(3,3))
else:
    print("UNSAT, try increasing nr of steps? or wrong input...")
