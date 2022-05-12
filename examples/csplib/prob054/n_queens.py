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

if __name__ == "__main__":

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    n_sols = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    solver = sys.argv[3] if len(sys.argv) > 3 else None


    queens = intvar(1, n, shape=n)

    # Constraints on columns and left/right diagonal
    model = Model([
        AllDifferent(queens),
        AllDifferent(queens - np.arange(n)),
        AllDifferent(queens + np.arange(n)),
    ])

    num_solutions = model.solveAll(solution_limit=n_sols,
                                   solver=solver,
                                   display = lambda : print(queens.value(),end="\n\n"))

    print("num_solutions:", num_solutions)