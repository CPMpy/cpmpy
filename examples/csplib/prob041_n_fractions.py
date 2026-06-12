"""
N-fractions problem in cpmpy.

Problem 041 on CSPlib
https://www.csplib.org/Problems/prob041/

Find distinct non-zero digits such that the following equation holds:
    A / BC + D / EF + G / HI = 1
Here, BC, EF, and HI are two-digit numbers formed by B and C, E and F,
and H and I, respectively.

Model from DCP-Bench-Open (https://github.com/DCP-Bench/DCP-Bench-Open/blob/main/dataset/csplib_041_n_fractions/csplib_041_n_fractions.cpmpy.py)
"""

from cpmpy import *


def n_fractions(n=9):
    x = intvar(1, n, shape=9, name="x")
    A, B, C, D, E, F, G, H, I = x

    D1 = intvar(1, n * n, name="D1")
    D2 = intvar(1, n * n, name="D2")
    D3 = intvar(1, n * n, name="D3")

    model = Model([AllDifferent(x),
                   D1 == 10 * B + C,
                   D2 == 10 * E + F,
                   D3 == 10 * H + I,
                   A * D2 * D3 + D * D1 * D3 + G * D1 * D2 == D1 * D2 * D3])

    return model, (x,)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", type=int, default=9, help="Range of digits (1..n)")

    n = parser.parse_args().n

    model, (x,) = n_fractions(n)

    if model.solve():
        A, B, C, D, E, F, G, H, I = x.value()
        print(f"A={A}, B={B}, C={C}, D={D}, E={E}, F={F}, G={G}, H={H}, I={I}")
        print(f"{A}/{10*B+C} + {D}/{10*E+F} + {G}/{10*H+I} = 1")
    else:
        raise ValueError("Model is unsatisfiable")
