"""
N-queens problem in CPMpy

CSPlib prob054

Problem description from the numberjack example:
The N-Queens problem is the problem of placing N queens on an N x N chess
board such that no two queens are attacking each other. A queen is attacking
another if it they are on the same row, same column, or same diagonal.

Here are some different approaches with different version of both
the constraints and how to solve and print all solutions.


This CPMpy model was written by Hakan Kjellerstrand (hakank@gmail.com)
See also my CPMpy page: http://hakank.org/cpmpy/

Modified by Ignace Bleukx
"""
import sys
import numpy as np
from cpmpy import *

def n_queens(n=16):

    queens = intvar(1, n, shape=n)

    # Constraints on columns and left/right diagonal
    model = Model([
        AllDifferent(queens),
        AllDifferent(queens - np.arange(n)),
        AllDifferent(queens + np.arange(n)),
    ])

    return model, (queens,)

if __name__ == "__main__":

    n = 16
    n_sols = 0 # find all solutions
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    if len(sys.argv) > 2:
        n_sols = int(sys.argv[2])

    model, (queens,) = n_queens(n)

    n_sols = model.solveAll(solution_limit=n_sols,
                            display = lambda : print(queens.value(),end="\n\n"))

    print("num_solutions:", n_sols)