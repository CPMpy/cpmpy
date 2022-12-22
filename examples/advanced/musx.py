"""
Deletion-based Minimum Unsatisfiable Subset (MUS) algorithm.

This is now part of the CPMpy tools!
"""

from cpmpy import *
from cpmpy.tools import mus

def main():
    x = intvar(-9, 9, name="x")
    y = intvar(-9, 9, name="y")
    m = Model(
        x < 0, 
        x < 1,
        x > 2,
        (x + y > 0) | (y < 0),
        (y >= 0) | (x >= 0),
        (y < 0) | (x < 0),
        (y > 0) | (x < 0),
        AllDifferent(x,y) # invalid for musx_assum
    )
    print(m)
    assert (m.solve() is False)

    mymus = mus(m.constraints, [])
    print("\nMUS:", mymus)


def musx(soft_constraints, hard_constraints=[], verbose=False):
    return mus(soft_constraints, hard_constraints)

def musx_pure(soft_constraints, hard_constraints=[], verbose=False):
    from cpmpy.tools.mus import mus_naive
    return mus_naive(soft_constraints, hard_constraints)

def musx_assum(soft_constraints, hard_constraints=[], verbose=False):
    return mus(soft_constraints, hard_constraints)


if __name__ == '__main__':
    main()
