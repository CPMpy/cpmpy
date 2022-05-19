"""
Steiner triplets in CPMpy.

Problem 044 on CSPlib

Model created by Ignce Bleukx
"""
from cpmpy import *
from cpmpy.expressions.utils import all_pairs

import numpy as np


def print_sol(sets):
    for s in sets:
        print(np.where(s)[0], end=" ")
    print()

if __name__ == "__main__":

    n = 15
    num_sols = 1
    assert n % 6 == 1 or n % 6 == 3, "N must be (1|3) modulo 6"

    n_sets = int(n * (n - 1) // 6)

    model = Model()

    # boolean representation of sets
    # sets[i,j] = true iff item j is part of set i
    sets = boolvar(shape=(n_sets, n), name="sets")

    # cardinality of set if 3
    # can be written cleaner, see issue #117
    # model += sum(sets, axis=0) == 3
    model += [sum(s) == 3 for s in sets]

    # cardinality of intersection <= 1
    for s1, s2 in all_pairs(sets):
        model += sum(s1 & s2) <= 1

    # symmetry breaking
    model += (sets[(0, 0)] == 1)

    model.solveAll(solver="pysat",
                   solution_limit=num_sols,
                   display=lambda : print_sol(sets.value()))