"""
Hadamard matrix Legendre pairs in CPMpy.
Problem 084 on CSPlib

Model created by Ignace Bleukx, ignace.bleukx@kuleuven.be
"""
import sys

import numpy as np
from cpmpy import *


def PAF(arr, s):
    return sum(arr * np.roll(arr,-s))

def hadmard_matrix(l=5):

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

    return model, (a,b)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-length", type=int, default=5, help="Length of sequence")

    l = parser.parse_args().length

    model, (a,b) = hadmard_matrix(l)

    if model.solve():
        print(f"A: {a.value()}")
        print(f"B: {b.value()}")
    else:
        raise ValueError("Model is unsatisfiable")
