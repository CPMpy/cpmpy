"""
    Minimizing autocorrelation of bitarray in CPMpy

    Problem 005 on CSPlib

    Model created by Ignace Bleukx, ignace.bleukx@kuleuven.be
"""
import argparse

import numpy as np

from cpmpy import *


def auto_correlation(n=16):

    # bitarray of length n
    arr = intvar(-1,1,shape=n, name="arr")

    model = Model()

    # exclude 0
    model += arr != 0

    # minimize sum of squares
    model.minimize(
        sum([PAF(arr,s) ** 2 for s in range(1,n)])
    )

    return model, (arr,)

# periodic auto correlation
def PAF(arr, s):
    # roll the array 's' indices
    return sum(arr * np.roll(arr,-s))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-length", nargs='?', type=int, default=16, help="Length of bitarray")

    length = parser.parse_args().length

    model, (arr,) = auto_correlation(length)

    if model.solve():
        # print using runlength notation
        arr = arr.value()
        pieces = np.split(arr, np.where(arr == -1)[0])

        print("Run length encoding of solution:")
        print("".join([str(len(p)) for p in pieces if len(p) != 0]))

    else:
        raise ValueError("Model is unsatisfiable")