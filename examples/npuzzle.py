#!/usr/bin/python3
"""
N-puzzle problem in CPMpy

Based on the state-space approach of 'lpcp21_p1_frog',
originally motivated by Hakan Kjellerstrand's
https://github.com/hakank/hakank/blob/master/minizinc/n_puzzle.mzn

A typical children toy, where a picture is divided in n+1 blocks and mangled,
and the goal is to trace the steps to the original picture.

It is a typical planning problem, here solved with CSP over a finite
planning horizon of `N` steps.
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
    [1,7,5]]) # 13 steps
"""
puzzle_start = np.array([
    [3,7,5],
    [1,6,4],
    [8,2,0]]) # 19 steps
puzzle_start = np.array([
    [8,3,4],
    [1,7,6],
    [2,5,0]]) # 25 steps
puzzle_start = np.array([
    [8,1,0],
    [3,4,7],
    [2,5,6]]) # 29 steps
"""
puzzle_end = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,0]])

def n_puzzle(puzzle_start, puzzle_end, N):
    # max nr steps to the solution
    print("Max steps:", N)
    m = Model()

    (dim,dim2) = puzzle_start.shape
    assert (dim == dim2), "puzzle needs square shape"
    n = dim*dim2 - 1 # e.g. an 8-puzzle

    # State of puzzle at every step
    x = intvar(0,n, shape=(N,dim,dim), name="x")

    # Start state constraint
    m += (x[0] == puzzle_start)

    # End state constraint
    m += (x[-1] == puzzle_end)

    # define neighbors = allowed moves for the '0'
    def neigh(i,j):
        # same, left,right, down,up, if within bounds
        for (rr, cc) in [(0,0),(-1,0),(1,0),(0,-1),(0,1)]:
            if 0 <= i+rr and i+rr < dim and 0 <= j+cc and j+cc < dim:
                yield (i+rr,j+cc)

    # Transition: define next based on prev + invariants
    def transition(m, prev_x, next_x):
        # for each position, determine its reachability
        for i in range(dim):
            for j in range(dim):
                m += (next_x[i,j] == 0).implies(any(prev_x[r,c] == 0 for r,c in neigh(i,j)))

        # Invariant: in each step, all cells are different
        m += AllDifferent(next_x)

        # Invariant: only the '0' position can move
        m += ((prev_x == 0) | (next_x == 0) | (prev_x == next_x))

    # apply transitions (0,1) (1,2) (2,3) ...
    for i in range(1, N):
        transition(m, x[i-1], x[i])

    return (m,x)

N = 20 # max nr steps
(m,x) = n_puzzle(puzzle_start, puzzle_end, N)
# Lets minimize the number of steps used...
is_sol = [all((x[i] == puzzle_end).flat) for i in range(N)]
# which means, maximize nr of late steps that are full sol
m.maximize( sum(i*is_sol[i] for i in range(N)) )

if m.solve() is False:
    print("UNSAT, try increasing nr of steps? or wrong input...")
else:
    for i in range(N):
        print("Step", i+1)
        print(x[i].value())
        if (x[i].value() != puzzle_end).sum() == 0:
            # all puzzle_end
            break
print(m.status())

"""
# previous visualisation helper function
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
"""
