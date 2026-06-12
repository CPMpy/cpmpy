"""
Magic hexagon problem in cpmpy.

Problem 023 on CSPlib
https://www.csplib.org/Problems/prob023/

A magic hexagon problem involves arranging the numbers 1 to 19 in a hexagonal pattern
such that the sum of the numbers in each of the 15 lines (rows and diagonals) is
equal to a magic constant, which is 38.

The hexagonal grid is structured as follows:
      A, B, C
     D, E, F, G
    H, I, J, K, L
     M, N, O, P
      Q, R, S

The goal is to find an assignment of numbers to the variables A through S that
satisfies all the sum constraints.

Model from DCP-Bench-Open (https://github.com/DCP-Bench/DCP-Bench-Open/blob/main/dataset/csplib_023_magic_hexagon/csplib_023_magic_hexagon.cpmpy.py)
"""

from cpmpy import *


def magic_hexagon(num_cells=19, magic_sum=38):
    LD = intvar(1, num_cells, shape=num_cells, name="LD")
    a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s = LD

    model = Model()

    # All numbers from 1 to 19 used exactly once
    model += AllDifferent(LD)

    # Rows (horizontal)
    model += (sum([a, b, c]) == magic_sum)
    model += (sum([d, e, f, g]) == magic_sum)
    model += (sum([h, i, j, k, l]) == magic_sum)
    model += (sum([m, n, o, p]) == magic_sum)
    model += (sum([q, r, s]) == magic_sum)

    # Diagonals (top-left to bottom-right)
    model += (sum([a, d, h]) == magic_sum)
    model += (sum([b, e, i, m]) == magic_sum)
    model += (sum([c, f, j, n, q]) == magic_sum)
    model += (sum([g, k, o, r]) == magic_sum)
    model += (sum([l, p, s]) == magic_sum)

    # Diagonals (top-right to bottom-left)
    model += (sum([c, g, l]) == magic_sum)
    model += (sum([b, f, k, p]) == magic_sum)
    model += (sum([a, e, j, o, s]) == magic_sum)
    model += (sum([d, i, n, r]) == magic_sum)
    model += (sum([h, m, q]) == magic_sum)

    return model, (LD,)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    model, (LD,) = magic_hexagon()

    if model.solve():
        vals = LD.value()
        print(f"  {vals[0]} {vals[1]} {vals[2]}")
        print(f" {vals[3]} {vals[4]} {vals[5]} {vals[6]}")
        print(f"{vals[7]} {vals[8]} {vals[9]} {vals[10]} {vals[11]}")
        print(f" {vals[12]} {vals[13]} {vals[14]} {vals[15]}")
        print(f"  {vals[16]} {vals[17]} {vals[18]}")
    else:
        raise ValueError("Model is unsatisfiable")
