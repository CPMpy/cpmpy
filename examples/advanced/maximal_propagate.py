"""
Example showing repeated solving for deriving the maximal consequence of set of constraints.

Iteratively finds a new solutions by forcing at least one variable to have a value not seen in any other solution.
"""

from cpmpy import *
from cpmpy.transformations.get_variables import get_variables


def maximal_propagate(constraints, solvername="ortools"):
    # For faster propagation, use incremental solver such as pysat, z3 or gurobi.
    vars = get_variables(constraints)
    visisted_domain = {var: set() for var in vars}

    solver = SolverLookup.get(solvername)
    solver += constraints

    while solver.solve():
        cons = False
        for var, values in visisted_domain.items():
            values.add(var.value())
            cons |= all(var != val for val in values)
        solver += cons
    # exhausted all possible domain values
    return visisted_domain


if __name__ == "__main__":

    x,y,z = [intvar(lb=0,ub=5, name=n) for n in "xyz"]

    model = Model([x < y, y != 3])
    print("Propagating constraints in model:", model, sep="\n", end="\n\n")

    possible_vals = maximal_propagate(model.constraints)

    for var, values in possible_vals.items():
        print(f"{var}: {sorted(values)}")
