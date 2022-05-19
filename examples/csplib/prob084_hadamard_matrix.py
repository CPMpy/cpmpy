"""
Hadamard matrix Legendre pairs in CPMpy.
Problem 084 on CSPlib

Model created by Ignace Bleukx
"""

import numpy as np
from cpmpy import *


def PAF(arr, s):
    return sum(arr * np.roll(arr,-s))


if __name__ == "__main__":

    l = 5
    m = int((l - 1) / 2)

    a = intvar(-1,1, shape=l, name="a")
    b = intvar(-1,1, shape=l, name="b")

    model = Model()

    model += a != 0 # exclude 0 from dom
    model += b != 0 # exclude 0 from dom

    model += sum(a) == 1
    model += sum(b) == 1

    for s in range(1,m+1):
        model += (PAF(a,s) + PAF(b,s)) == -2

    if model.solve():
        print(f"{a.value()= }")
        print(f"{b.value()= }")
    else:
        print("Model is unsatisfiable")




