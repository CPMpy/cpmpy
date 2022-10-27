"""
Golomb's ruler problem in cpmpy.

Problem 006 on CSPlib
https://www.csplib.org/Problems/prob006/

A Golomb ruler is a set of integers (marks) a(1) < ...  < a(n) such
that all the differences a(i)-a(j) (i > j) are distinct.  Clearly we
may assume a(1)=0.  Then a(n) is the length of the Golomb ruler.
For a given number of marks, n, we are interested in finding the
shortest Golomb rulers.  Such rulers are called optimal.

Also, see:
- https://en.wikipedia.org/wiki/Golomb_ruler
- http://www.research.ibm.com/people/s/shearer/grule.html


Model created by Hakan Kjellerstrand, hakank@hakank.com
See also my cpmpy page: http://www.hakank.org/cpmpy/

Modified by Ignace Bleukx, ignace.bleukx@kuleuven.be
"""

from cpmpy import *

def golomb(size=10):

    marks = intvar(0, size*size, shape=size, name="marks")

    model = Model()
    # first mark is 0
    model += (marks[0] == 0)
    # marks must be increasing
    model += marks[:-1] < marks[1:]

    # golomb constraint
    diffs = [marks[j] - marks[i] for i in range(0, size-1)
                                 for j in range(i+1, size)]
    model += AllDifferent(diffs)

    # Symmetry breaking
    model += (marks[size - 1] - marks[size - 2] > marks[1] - marks[0])
    model += (diffs[0] < diffs[size - 1])

    # find optimal ruler
    model.minimize(marks[-1])

    return model, (marks,)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-size", type=int, default=10, help="Size of the ruler")

    size = parser.parse_args().size

    model, (marks, ) = golomb(size)

    if model.solve():
        print(marks.value())
    else:
        raise ValueError("Model is unsatisfiable")


