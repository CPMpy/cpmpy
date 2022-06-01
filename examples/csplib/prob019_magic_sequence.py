"""
Magic sequence in cpmpy.

Problem 019 on CSPlib
https://www.csplib.org/Problems/prob019/

A magic sequence of length n is a sequence of integers x0 . . xn-1 between
0 and n-1, such that for all i in 0 to n-1, the number i occurs exactly xi
times in the sequence. For instance, 6,2,1,0,0,0,1,0,0,0 is a magic sequence
since 0 occurs 6 times in it, 1 occurs twice, ...

Model created by Hakan Kjellerstrand, hakank@hakank.com
See also my cpmpy page: http://www.hakank.org/cpmpy/

Modified by Ignace Bleukx, ignace.bleukx@kuleuven.be
"""
import sys
import numpy as np
from cpmpy import *

def magic_sequence(n=10):
    print("n:", n)
    model = Model()

    x = intvar(0, n - 1, shape=n, name="x")

    # constraints
    for i in range(n):
        model += x[i] == sum(x == i)

    # speedup search
    model += sum(x) == n
    model += sum(x * np.arange(n)) == n

    return model, (x,)

if __name__ == "__main__":

    n = 10
    if len(sys.argv) > 1:
        n = int(sys.argv[1])

    model, (x,) = magic_sequence(n)

    if model.solve():
        print(f"x: {x.value()}")
    else:
        print("Model is unsatisviable)")




