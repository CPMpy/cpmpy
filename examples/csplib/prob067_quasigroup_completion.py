"""
Quasigroup completion problem in cpmpy.

Problem 067 on CSPlib
https://www.csplib.org/Problems/prob067/

An order m quasigroup is a Latin square of size m. That is, an m x m
multiplication table in which each element from 1 to m occurs exactly once
in every row and every column. This problem asks to complete a partially
filled quasigroup.

Model from DCP-Bench-Open (https://github.com/DCP-Bench/DCP-Bench-Open/blob/main/dataset/csplib_067_quasigroup_completion/csplib_067_quasigroup_completion.cpmpy.py)
"""

import cpmpy as cp
import numpy as np


def quasigroup_completion(N=5, start=None):
    if start is None:
        start = [[1, 0, 0, 0, 0],
                 [0, 2, 0, 0, 0],
                 [0, 0, 3, 0, 0],
                 [0, 0, 0, 4, 0],
                 [0, 0, 0, 0, 5]]

    puzzle = cp.intvar(1, n, shape=(n, n), name="puzzle")

    model = cp.Model()

    # Constraints
    # 1. Pre-fill the grid with the starting values.
    # We iterate through the start matrix and add a constraint for each non-zero cell.
    start_np = np.array(start)
    for i in range(N):
        for j in range(N):
            if start_np[i, j] != 0:
                model += puzzle[i, j] == start_np[i, j]

    model += [cp.AllDifferent(row) for row in puzzle]
    model += [cp.AllDifferent(col) for col in puzzle.T]

    return model, (puzzle,)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", type=int, default=5, help="Order of the quasigroup")

    n = parser.parse_args().n

    model, (puzzle,) = quasigroup_completion(n)

    if model.solve():
        for row in puzzle.value():
            print(" ".join(str(int(v)) for v in row))
    else:
        raise ValueError("Model is unsatisfiable")
