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
    D = 9999

    primes = [i for i in range(2, D + 1) if is_prime(i)]
    squares = [k * k for k in range(1, math.isqrt(D) + 1)]

    Z = -1
    B = -2
    Valid = [[Z, Z, Z, Z, B, Z, Z, Z, Z],
             [Z, Z, B, Z, Z, Z, B, Z, Z],
             [Z, B, Z, Z, B, Z, Z, B, Z],
             [Z, Z, Z, Z, B, Z, Z, Z, Z],
             [B, Z, B, B, B, B, B, Z, B],
             [Z, Z, Z, Z, B, Z, Z, Z, Z],
             [Z, B, Z, Z, B, Z, Z, B, Z],
             [Z, Z, B, Z, Z, Z, B, Z, Z],
             [Z, Z, Z, Z, B, Z, Z, Z, Z]]

    M = cp.intvar(0, 9, shape=(n, n), name="M")

    model = cp.Model()

    for i in range(n):
        for j in range(n):
            if Valid[i][j] == B:
                model += (M[i, j] == 0)

    # Across variables
    A1 = cp.intvar(0, D, name="A1")
    A4 = cp.intvar(0, D, name="A4")
    A7 = cp.intvar(0, D, name="A7")
    A8 = cp.intvar(0, D, name="A8")
    A9 = cp.intvar(0, D, name="A9")
    A10 = cp.intvar(0, D, name="A10")
    A11 = cp.intvar(0, D, name="A11")
    A13 = cp.intvar(0, D, name="A13")
    A15 = cp.intvar(0, D, name="A15")
    A17 = cp.intvar(0, D, name="A17")
    A20 = cp.intvar(0, D, name="A20")
    A23 = cp.intvar(0, D, name="A23")
    A24 = cp.intvar(0, D, name="A24")
    A25 = cp.intvar(0, D, name="A25")
    A27 = cp.intvar(0, D, name="A27")
    A28 = cp.intvar(0, D, name="A28")
    A29 = cp.intvar(0, D, name="A29")
    A30 = cp.intvar(0, D, name="A30")

    # Down variables
    D1 = cp.intvar(0, D, name="D1")
    D2 = cp.intvar(0, D, name="D2")
    D3 = cp.intvar(0, D, name="D3")
    D4 = cp.intvar(0, D, name="D4")
    D5 = cp.intvar(0, D, name="D5")
    D6 = cp.intvar(0, D, name="D6")
    D10 = cp.intvar(0, D, name="D10")
    D12 = cp.intvar(0, D, name="D12")
    D14 = cp.intvar(0, D, name="D14")
    D16 = cp.intvar(0, D, name="D16")
    D17 = cp.intvar(0, D, name="D17")
    D18 = cp.intvar(0, D, name="D18")
    D19 = cp.intvar(0, D, name="D19")
    D20 = cp.intvar(0, D, name="D20")
    D21 = cp.intvar(0, D, name="D21")
    D22 = cp.intvar(0, D, name="D22")
    D26 = cp.intvar(0, D, name="D26")
    D28 = cp.intvar(0, D, name="D28")

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

    return model, (M,)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    model, (M,) = crossfigures()

    if model.solve():
        for row in M.value():
            print(" ".join(str(int(cell)) for cell in row))
    else:
        raise ValueError("Model is unsatisfiable")
