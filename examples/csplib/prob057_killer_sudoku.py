"""
Killer sudoku problem in cpmpy.

Problem 057 on CSPlib
https://www.csplib.org/Problems/prob057/

Killer sudoku (also killer su doku, sumdoku, sum doku, addoku, or
samunamupure) is a puzzle that combines elements of sudoku and kakuro.
Despite the name, the simpler killer sudokus can be easier to solve
than regular sudokus, depending on the solver's skill at mental arithmetic;
the hardest ones, however, can take hours to crack.

The objective is to fill the grid with numbers from 1 to 9 in a way that
the following conditions are met:

  * Each row, column, and nonet contains each number exactly once.
  * The sum of all numbers in a cage must match the small number printed
    in its corner.
  * No number appears more than once in a cage. (This is the standard rule
    for killer sudokus, and implies that no cage can include more
    than 9 cells.)

Model from DCP-Bench-Open (https://github.com/DCP-Bench/DCP-Bench-Open/blob/main/dataset/csplib_057_killer_sudoku/csplib_057_killer_sudoku.cpmpy.py)
"""

import cpmpy as cp


def killer_sudoku(n=9, problem=None):
    if problem is None:
        problem = [[3, [[1, 1], [1, 2]]], [15, [[1, 3], [1, 4], [1, 5]]],
                   [22, [[1, 6], [2, 5], [2, 6], [3, 5]]], [4, [[1, 7], [2, 7]]],
                   [16, [[1, 8], [2, 8]]], [15, [[1, 9], [2, 9], [3, 9], [4, 9]]],
                   [25, [[2, 1], [2, 2], [3, 1], [3, 2]]], [17, [[2, 3], [2, 4]]],
                   [9, [[3, 3], [3, 4], [4, 4]]], [8, [[3, 6], [4, 6], [5, 6]]],
                   [20, [[3, 7], [3, 8], [4, 7]]], [6, [[4, 1], [5, 1]]],
                   [14, [[4, 2], [4, 3]]], [17, [[4, 5], [5, 5], [6, 5]]],
                   [17, [[4, 8], [5, 7], [5, 8]]], [13, [[5, 2], [5, 3], [6, 2]]],
                   [20, [[5, 4], [6, 4], [7, 4]]], [12, [[5, 9], [6, 9]]],
                   [27, [[6, 1], [7, 1], [8, 1], [9, 1]]],
                   [6, [[6, 3], [7, 2], [7, 3]]], [20, [[6, 6], [7, 6], [7, 7]]],
                   [6, [[6, 7], [6, 8]]], [10, [[7, 5], [8, 4], [8, 5], [9, 4]]],
                   [14, [[7, 8], [7, 9], [8, 8], [8, 9]]], [8, [[8, 2], [9, 2]]],
                   [16, [[8, 3], [9, 3]]], [15, [[8, 6], [8, 7]]],
                   [13, [[9, 5], [9, 6], [9, 7]]], [17, [[9, 8], [9, 9]]]]

    x = cp.intvar(1, n, shape=(n, n), name="x")

    model = cp.Model()

    model += [cp.AllDifferent(row) for row in x]
    model += [cp.AllDifferent(col) for col in x.transpose()]

    for i in range(2):
        for j in range(2):
            cell = [x[r, c] for r in range(i * 3, i * 3 + 3) for c in range(j * 3, j * 3 + 3)]
            model += [cp.AllDifferent(cell)]

    for (res, segment) in problem:
        cage = [x[i[0] - 1, i[1] - 1] for i in segment]
        model += cp.sum(cage) == res
        model += cp.AllDifferent(cage)

    return model, (x,)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    model, (x,) = killer_sudoku()

    if model.solve():
        for row in x.value():
            print(" ".join(str(int(v)) for v in row))
    else:
        raise ValueError("Model is unsatisfiable")
