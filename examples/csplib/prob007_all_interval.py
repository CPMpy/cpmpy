"""
All interval problem in cpmpy.

CSPLib problem number 007
https://www.csplib.org/Problems/prob007/


Given the twelve standard pitch-classes (c, c , d, ...), represented by
numbers 0,1,...,11, find a series in which each pitch-class occurs exactly
once and in which the musical intervals between neighbouring notes cover
the full set of intervals from the minor second (1 semitone) to the major
seventh (11 semitones). That is, for each of the intervals, there is a
pair of neigbhouring pitch-classes in the series, between which this
interval appears. The problem of finding such a series can be easily
formulated as an instance of a more general arithmetic problem on Z_n,
the set of integer residues modulo n. Given n in N, find a vector
s = (s_1, ..., s_n), such that (i) s is a permutation of
Z_n = {1,2,...,n}; and (ii) the interval vector
v = (|s_2-s_1|, |s_3-s_2|, ... |s_n-s_{n-1}|) is a permutation of
Z_n-{0} = {1,2,...,n-1}. A vector v satisfying these conditions is
called an all-interval series of size n; the problem of finding such
a series is the all-interval series problem of size n. We may also be
interested in finding all possible series of a given size.


Model created by Hakan Kjellerstrand, hakank@hakank.com
See also my cpmpy page: http://www.hakank.org/cpmpy/

Modified by Ignace Bleukx

"""
import argparse

from cpmpy import *
import numpy as np


def all_interval(n=12):

  # Create the solver.
  model = Model()

  # declare variables
  x = intvar(1,n,shape=n,name="x")
  diffs = intvar(1,n-1,shape=n-1,name="diffs")

  # constraints
  model += [AllDifferent(x),
            AllDifferent(diffs)]

  model += diffs == np.abs(x[1:] - x[:-1])

  # symmetry breaking
  model += [x[0] < x[n - 1]]
  model += [diffs[0] < diffs[1]]

  return model, (x, diffs)

def print_solution(x, diffs):
    print(f"x:    {x.value()}")
    print(f"diffs: {diffs.value()}")
    print()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", type=int,help="Length of array", default=12, required=False)
    parser.add_argument("-num_sols", type=int, help="Number of solutions to find, find all by default", default=0, required=False)

    args = parser.parse_args()

    model, (x, diffs) = all_interval(args.n)
    found_n = model.solveAll(solution_limit=args.num_sols,
                             display=lambda: print_solution(x, diffs))

    print(f"Found {found_n} solutions")