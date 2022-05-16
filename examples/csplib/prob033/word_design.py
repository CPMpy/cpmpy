"""
Word design in CPMpy

Problem 033 on CSPlib

Model created by Ignace Bleukx
"""
import sys

from cpmpy import *
import numpy as np


if __name__ == "__main__":

    n = 112
    print(f"Attempting to find at most {n} words")

    A,C,G,T = 1,2,3,4

    # words[i,j] is the j'th letter of the i'th word
    words = intvar(A,T,shape=(n,8), name="words")

    model = Model()

    # 4 symbols from {C,G}
    for w in words:
        model += sum((w == C) | (w == G)) >= 4

    for y in words:
        y_c = 5 - y  # Watson-Crick complement
        for x in words:
            # x^R and y^C differ in at least 4 positions
            x_r = x[::-1] # reversed x
            model += sum((x_r != y_c)) >= 4

    # break symmetry
    for r in range(n-1):
        b = boolvar(n+1)
        model += b[0] == 1
        model += b == ((words[r] <= words[r + 1]) &
                       ((words[r] < words[r + 1]) | b[1:] == 1))
        model += b[-1] == 0

    if model.solve():
        map = np.array(["A","C","G","T"])
        for word in words.value():
            if np.any(word != 0):
                print("".join(map[word-1]))

    else:
        print("Model is unsatisfiable")