"""
Langford's number problem in cpmpy.

Langford's number problem (CSP lib problem 24)
http://www.csplib.org/prob/prob024/

Arrange 2 sets of positive integers 1..k to a sequence,
such that, following the first occurence of an integer i,
each subsequent occurrence of i, appears i+1 indices later
than the last.
For example, for k=4, a solution would be 41312432


Model created by Hakan Kjellerstrand, hakank@hakank.com
See also my cpmpy page: http://www.hakank.org/cpmpy/

Modified by Ignace Bleukx
"""
import sys
from cpmpy import *
import numpy as np

def langford(k=8):
    model = Model()

    if not (k % 4 == 0 or k % 4 == 3):
        print("There is no solution for K unless K mod 4 == 0 or K mod 4 == 3")
        return

    # variables
    position = intvar(0, 2 * k - 1, shape=2 * k, name="position")
    solution = intvar(1, k, shape=2 * k, name="solution")

    # constraints
    model += [AllDifferent(position)]

    # can be written without for-loop, see issue #117 on github
    # i = np.arange(1,k+1)
    # model += position[i + k -1] == (position[i - 1] + i  + 1)
    # model += solution[position[i-1]] == i
    # model += solution[position[k+i-1]] == i

    for i in range(1, k + 1):
        model += [position[i + k - 1] == position[i - 1] + i + 1]
        model += [i == solution[position[i - 1]]]
        model += [i == solution[position[k + i - 1]]]

    # symmetry breaking
    model += [solution[0] < solution[2 * k - 1]]

    return model, (position, solution)


def print_solution(position, solution):
    print(f"position: {position.value()}")
    print(f"solution: {solution.value()}")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-k", type=int, default=8, help="Number of integers")
    parser.add_argument("--solution_limit", type=int, default=0, help="Number of solutions to search for, find all by default")

    args = parser.parse_args()

    model, (position, solution) = langford(args.k)

    num_sols = model.solveAll(solution_limit=args.solution_limit,
                              display=lambda: print_solution(position, solution))

    if num_sols == 0:
        raise ValueError("Model is unsatsifiable")
