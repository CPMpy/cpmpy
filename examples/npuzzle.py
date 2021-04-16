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
    [1,7,5]]).reshape(-1) # 12 steps
"""
puzzle_start = np.array([
    [3,7,5],
    [1,6,4],
    [8,2,0]]).reshape(-1) # 18 steps
puzzle_start = np.array([
    [8,3,4],
    [1,7,6],
    [2,5,0]]).reshape(-1) # 24 steps
puzzle_start = np.array([
    [8,1,0],
    [3,4,7],
    [2,5,6]]).reshape(-1) # 28 steps
"""
puzzle_end = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,0]]).reshape(-1)
n = puzzle_start.shape[0] # flat length
dim = int(math.sqrt(n))
assert (dim*dim == n), "non-square matrix??"

# max nr steps to the solution
num_sols = 30

# Generate the allowed moves (including to same position)
# such that (move_from, move_to) in valid_table
valid_table = []
def pos(r,c):
    return r*dim+c
for r in range(dim):
    for c in range(dim):
        # same, up, down, left, right
        for (rr, cc) in [(0,0),(-1,0),(1,0),(0,-1),(0,1)]: 
            if 0 <= r+rr and r+rr < dim and 0 <= c+cc and c+cc < dim:
                valid_table.append([pos(r,c), pos(r+rr,c+cc)])

# Variables
x = IntVar(0,n-1, shape=(num_sols,n))

# the moves, index in puzzle
move_from = IntVar(0,n-1, shape=(num_sols)) # is move_to[i-1]...
move_to = IntVar(0,n-1, shape=(num_sols))

# is this row the solution?
check = BoolVar(shape=num_sols)

# index of first solution
check_ix = IntVar(0,num_sols)


m = Model(minimize=check_ix)

def same(x, y):
    return all(x == y)

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
    for i in range(1,num_sols) for j in range(n)
]


# require valid moves
m += [ Table( (move_from[i],move_to[i]), valid_table) for i in range(num_sols)]


# check whether this is the end solution
m += [ check[i] == same(x[i,:], puzzle_end) for i in range(num_sols)]

# once the solution is found, keep it
m += [ check[i] >= check[i-1] for i in range(1,num_sols) ]

# index of a solution (when minimized, index of first solution)
m += [ check[check_ix] == 1 ]


# visualisation helper function
def visu_npuzzle(xval, pos_from, pos_to):
    out = ""
    for r in range(dim):
        for c in range(dim):
            xpos = r*dim+c
            if xpos == pos_from or xpos == pos_to:
                out += f"{xval[xpos]}* "
            else:
                out += f"{xval[xpos]}  "
        out += '\n'
    print(out)

if m.solve() is False:
    print("UNSAT, try increasing nr of steps? or wrong input...")
else:
    for i in range(check_ix.value()+1):
        visu_npuzzle(x[i,:].value(), move_from[i].value(), move_to[i].value())
    print(f"Found in {check_ix.value()} steps")
print(m.status())
