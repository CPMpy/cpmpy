"""
Example showing repeated solving for deriving the maximal consequence of set of constraints.

Iteratively finds a new solutions by forcing at least one variable to have a value not seen in any other solution.
"""

from cpmpy import *
from cpmpy.tools.maximal_propagate import maximal_propagate


if __name__ == "__main__":
    x,y,z = [intvar(lb=0,ub=5, name=n) for n in "xyz"]

    model = Model([x < y, y != 3])
    print("Propagating constraints in model:", model, sep="\n", end="\n\n")

    possible_vals = maximal_propagate(model.constraints)

    for var, values in possible_vals.items():
        print(f"{var}: {sorted(values)}")
