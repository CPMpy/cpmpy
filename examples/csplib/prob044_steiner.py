"""
Steiner triplets in CPMpy.

Problem 044 on CSPlib
csplib.org/Problems/prob044/

The ternary Steiner problem of order n consists of finding a set of n*(n−1)/6 triples of distinct integer elements in
{1,…,n} such that any two triples have at most one common element. It is a hypergraph problem coming from combinatorial
mathematics [luneburg1989tools] where n modulo 6 has to be equal to 1 or 3 [lindner2011topics].
One possible solution for n=7 is {{1, 2, 3}, {1, 4, 5}, {1, 6, 7}, {2, 4, 6}, {2, 5, 7}, {3, 4, 7}, {3, 5, 6}}.
The solution contains 7*(7−1)/6=7 triples.



Model created by Ignace Bleukx, ignace.bleukx@kuleuven.be
"""
import numpy as np

from cpmpy import *
from cpmpy.expressions.utils import all_pairs

def steiner(n=15):
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

    return model, (sets,)

def print_sol(sets):
    for s in sets.value():
        print(np.where(s)[0], end=" ")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-num_sets", type=int, default=15, help="Number of sets")
    parser.add_argument("--solution_limit", type=int, default=0, help="Number of solutions to find, find all by default")

    args = parser.parse_args()

    model, (sets,) = steiner(args.num_sets)
    n_sol = model.solveAll(solver="pysat:minisat22",
                           solution_limit=args.solution_limit,
                           display=lambda : print_sol(sets))

    if n_sol == 0:
        raise ValueError("Model is unsatisfiable")
    else:
        print(f"Found {n_sol} solutions")