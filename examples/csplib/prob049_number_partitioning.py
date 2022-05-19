"""
Problem 049 on CSPLib

Adapted from pycsp3 implementation: https://raw.githubusercontent.com/xcsp3team/pycsp3/master/problems/csp/academic/NumberPartitioning.py

Modified by Ignace Bleukx
"""
import sys
import numpy as np
from cpmpy import *


if __name__ == "__main__":
    if len(sys.argv) > 1:
        n = sys.argv[1]
    else:
        n = 8

    assert n % 2 == 0, "The value of n must be even"

    # x[i] is the ith value of the first set
    x = intvar(1, n, shape=n // 2)

    # y[i] is the ith value of the second set
    y = intvar(1, n, shape=n // 2)

    model = Model()

    model += AllDifferent(np.append(x,y))

    # sum of numbers is equal in both sets
    model += sum(x) == sum(y)

    # sum of squares is equal in both sets
    model += sum(x ** 2) == sum(y ** 2)

    # break symmetry
    model += x[:-1] <= x[1:]
    model += y[:-1] <= x[1:]

    if model.solve():
        print(f"x: {x.value()}")
        print(f"y: {y.value()}")

    else:
        print("Model is unsatisfiable")