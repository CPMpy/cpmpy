"""
    Minimizing autocorrelation of bitarray in CPMpy

    Problem 005 on CSPlib

    Model created by Ignace Bleukx
"""


from cpmpy import *
import numpy as np


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
    return sum(arr * np.roll(arr,-s))


if __name__ == "__main__":

    n = 16
    model, arr = auto_correlation(n)

    if model.solve():
        # print using runlength notation
        arr = arr.value()
        pieces = np.split(arr, np.where(arr == -1)[0])

        print("Run length encoding of solution:")
        print("".join([str(len(p)) for p in pieces if len(p) != 0]))

    else:
        print("Model is unsatisfiable")