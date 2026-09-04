"""
Crossfigures puzzle in cpmpy.

Problem 021 on CSPlib
https://www.csplib.org/Problems/prob021/

Crossfigures are the numerical equivalent of crosswords. You have a grid and some
clues with numerical answers to place on this grid. Clues come in several different
forms (for example: Across 1. 25 across times two, 2. five dozen, 5. a square number,
10. prime, 14. 29 across times 21 down ...).

Here is the specific problem:

  1  2  3  4  5  6  7  8  9
  ---------------------------
  1  2  _  3  X  4  _  5  6    1
  7  _  X  8  _  _  X  9  _    2
  _  X  10 _  X  11 12 X  _    3
  13 14 _  _  X  15 _  16 _    4
  X  _  X  X  X  X  X  _  X    5
  17 _  18 19 X  20 21 _ 22    6
  _  X  23 _  X  24 _  X  _    7
  25 26 X  27 _  _  X  28 _    8
  29 _  _  _  X  30 _  _  _    9

Here are the clues:

    #  Across
    #  1 27 across times two
    #  4 4 down plus seventy-one
    #  7 18 down plus four
    #  8 6 down divided by sixteen
    #  9 2 down minus eighteen
    # 10 Dozen in six gross
    # 11 5 down minus seventy
    # 13 26 down times 23 across
    # 15 6 down minus 350
    # 17 25 across times 23 across
    # 20 A square number
    # 23 A prime number
    # 24 A square number
    # 25 20 across divided by seventeen
    # 27 6 down divided by four
    # 28 Four dozen
    # 29 Seven gross
    # 30 22 down plus 450

    # Down
    #
    #  1 1 across plus twenty-seven
    #  2 Five dozen
    #  3 30 across plus 888
    #  4 Two times 17 across
    #  5 29 across divided by twelve
    #  6 28 across times 23 across
    # 10 10 across plus four
    # 12 Three times 24 across
    # 14 13 across divided by sixteen
    # 16 28 down times fifteen
    # 17 13 across minus 399
    # 18 29 across divided by eighteen
    # 19 22 down minus ninety-four
    # 20 20 across minus nine
    # 21 25 across minus fifty-two
    # 22 20 down times six
    # 26 Five times 24 across
    # 28 21 down plus twenty-seven

Model from DCP-Bench-Open (https://github.com/DCP-Bench/DCP-Bench-Open/blob/main/dataset/csplib_021_crossfigures/csplib_021_crossfigures.cpmpy.py)
"""

import math
import cpmpy as cp
import numpy as np


def is_prime(n):
    """Check if n is a prime number."""
    if n < 2: return False
    if n == 2: return True
    if not n & 1:
        return False
    for i in range(3, 1 + int(math.sqrt(n)), 2):
        if n % i == 0:
            return False
    return True


def to_num(a, n, base):
    """Constrain digit list a to represent number n in the given base."""
    tlen = len(a)
    return n == cp.sum([(base ** (tlen - i - 1)) * a[i] for i in range(tlen)])


def across(Matrix, Across, Len, Row, Col):
    """Link an across clue variable to its digit cells in the grid."""
    Row -= 1
    Col -= 1
    digits = [Matrix[Row, Col + i] for i in range(Len)]
    return [to_num(digits, Across, 10)]


def down(Matrix, Down, Len, Row, Col):
    """Link a down clue variable to its digit cells in the grid."""
    Row -= 1
    Col -= 1
    digits = [Matrix[Row + i, Col] for i in range(Len)]
    return [to_num(digits, Down, 10)]


def crossfigures():
    n = 9

    primes = [i for i in range(2, 10000) if is_prime(i)]
    squares = [k * k for k in range(1, math.isqrt(9999) + 1)]

    Z = '_'
    B = 'X'
    valid = np.array([
        [Z, Z, Z, Z, B, Z, Z, Z, Z],
        [Z, Z, B, Z, Z, Z, B, Z, Z],
        [Z, B, Z, Z, B, Z, Z, B, Z],
        [Z, Z, Z, Z, B, Z, Z, Z, Z],
        [B, Z, B, B, B, B, B, Z, B],
        [Z, Z, Z, Z, B, Z, Z, Z, Z],
        [Z, B, Z, Z, B, Z, Z, B, Z],
        [Z, Z, B, Z, Z, Z, B, Z, Z],
        [Z, Z, Z, Z, B, Z, Z, Z, Z]
    ])

    M = cp.intvar(0, 9, shape=(n, n), name="M")

    model = cp.Model()
    model += M[valid == B] == 0

    def clue_var(name, length):
        """Create a clue variable whose bounds match its number of grid cells."""
        return cp.intvar(10 ** (length - 1), 10 ** length - 1, name=name)

    # Across variables
    A1 = clue_var("A1", 4)
    A4 = clue_var("A4", 4)
    A7 = clue_var("A7", 2)
    A8 = clue_var("A8", 3)
    A9 = clue_var("A9", 2)
    A10 = clue_var("A10", 2)
    A11 = clue_var("A11", 2)
    A13 = clue_var("A13", 4)
    A15 = clue_var("A15", 4)
    A17 = clue_var("A17", 4)
    A20 = clue_var("A20", 4)
    A23 = clue_var("A23", 2)
    A24 = clue_var("A24", 2)
    A25 = clue_var("A25", 2)
    A27 = clue_var("A27", 3)
    A28 = clue_var("A28", 2)
    A29 = clue_var("A29", 4)
    A30 = clue_var("A30", 4)

    # Down variables
    D1 = clue_var("D1", 4)
    D2 = clue_var("D2", 2)
    D3 = clue_var("D3", 4)
    D4 = clue_var("D4", 4)
    D5 = clue_var("D5", 2)
    D6 = clue_var("D6", 4)
    D10 = clue_var("D10", 2)
    D12 = clue_var("D12", 2)
    D14 = clue_var("D14", 3)
    D16 = clue_var("D16", 3)
    D17 = clue_var("D17", 4)
    D18 = clue_var("D18", 2)
    D19 = clue_var("D19", 4)
    D20 = clue_var("D20", 4)
    D21 = clue_var("D21", 2)
    D22 = clue_var("D22", 4)
    D26 = clue_var("D26", 2)
    D28 = clue_var("D28", 2)

    # Set up matrix-digit constraints
    model += (across(M, A1, 4, 1, 1))
    model += (across(M, A4, 4, 1, 6))
    model += (across(M, A7, 2, 2, 1))
    model += (across(M, A8, 3, 2, 4))
    model += (across(M, A9, 2, 2, 8))
    model += (across(M, A10, 2, 3, 3))
    model += (across(M, A11, 2, 3, 6))
    model += (across(M, A13, 4, 4, 1))
    model += (across(M, A15, 4, 4, 6))
    model += (across(M, A17, 4, 6, 1))
    model += (across(M, A20, 4, 6, 6))
    model += (across(M, A23, 2, 7, 3))
    model += (across(M, A24, 2, 7, 6))
    model += (across(M, A25, 2, 8, 1))
    model += (across(M, A27, 3, 8, 4))
    model += (across(M, A28, 2, 8, 8))
    model += (across(M, A29, 4, 9, 1))
    model += (across(M, A30, 4, 9, 6))

    model += (down(M, D1, 4, 1, 1))
    model += (down(M, D2, 2, 1, 2))
    model += (down(M, D3, 4, 1, 4))
    model += (down(M, D4, 4, 1, 6))
    model += (down(M, D5, 2, 1, 8))
    model += (down(M, D6, 4, 1, 9))
    model += (down(M, D10, 2, 3, 3))
    model += (down(M, D12, 2, 3, 7))
    model += (down(M, D14, 3, 4, 2))
    model += (down(M, D16, 3, 4, 8))
    model += (down(M, D17, 4, 6, 1))
    model += (down(M, D18, 2, 6, 3))
    model += (down(M, D19, 4, 6, 4))
    model += (down(M, D20, 4, 6, 6))
    model += (down(M, D21, 2, 6, 7))
    model += (down(M, D22, 4, 6, 9))
    model += (down(M, D26, 2, 8, 2))
    model += (down(M, D28, 2, 8, 8))

    # Across clues
    model += (A1 == 2 * A27)
    model += (A4 == D4 + 71)
    model += (A7 == D18 + 4)
    model += (16 * A8 == D6)
    model += (A9 == D2 - 18)
    model += (12 * A10 == 6 * 144)
    model += (A11 == D5 - 70)
    model += (A13 == D26 * A23)
    model += (A15 == D6 - 350)
    model += (A17 == A25 * A23)
    model += cp.any(A20 == s for s in squares)
    model += cp.any(A23 == p for p in primes)
    model += cp.any(A24 == s for s in squares)
    model += (17 * A25 == A20)
    model += (4 * A27 == D6)
    model += (A28 == 4 * 12)
    model += (A29 == 7 * 144)
    model += (A30 == D22 + 450)

    # Down clues
    model += (D1 == A1 + 27)
    model += (D2 == 5 * 12)
    model += (D3 == A30 + 888)
    model += (D4 == 2 * A17)
    model += (12 * D5 == A29)
    model += (D6 == A28 * A23)
    model += (D10 == A10 + 4)
    model += (D12 == A24 * 3)
    model += (16 * D14 == A13)
    model += (D16 == 15 * D28)
    model += (D17 == A13 - 399)
    model += (18 * D18 == A29)
    model += (D19 == D22 - 94)
    model += (D20 == A20 - 9)
    model += (D21 == A25 - 52)
    model += (D22 == 6 * D20)
    model += (D26 == 5 * A24)
    model += (D28 == D21 + 27)

    return model, (M, valid)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    model, (M, layout) = crossfigures()

    if model.solve():
        values = M.value()

        for i, row in enumerate(values):
            print(" ".join('X' if layout[i, j] == 'X' else str(v) for j, v in enumerate(row)))
    else:
        raise ValueError("Model is unsatisfiable")
