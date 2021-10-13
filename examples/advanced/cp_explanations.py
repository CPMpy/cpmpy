from cpmpy.solvers.ortools import CPM_ortools
from cpmpy.solvers.pysat import CPM_pysat
from cpmpy.expressions.core import Operator
from cpmpy import *
from cpmpy.transformations.flatten_model import flatten_constraint
from cpmpy.transformations.get_variables import get_variables, get_variables_model

import numpy as np

def main(verbose=False):
    weights = [10, 10, 10, 1, 1, 40, 20, 20, 20, 1]
    x = intvar(-9, 9, name="x")
    y = intvar(-9, 9, name="y")
    m = Model(
        x < 0, 
        x < 1,
        x > 2,
        x == 4,
        y == 4, 
        (x + y > 0) | (y < 0),
        (y >= 0) | (x >= 0),
        (y < 0) | (x < 0),
        (y > 0) | (x < 0),
        AllDifferent(x,y) # invalid for musx_assum
    )
    assert (m.solve() is False)

    print(m)
    print("\nStart MUS search:")
    mus = ocus(m.constraints, weights, [], verbose=verbose)
    print("MUS:", mus)

def ocus(soft_constraints, soft_weights, hard_constraints=[], solver='ortools', verbose=False):
    """
        Hitting set based weighted-MUS for CP

        Each constraint is an arbitrary CPMpy expression, so it can
        also be sublists of constraints (e.g. constraint groups),
        contain aribtrary nested expressions, global constraints, etc.

        Will first check which soft contraints support reification,
        and use ocus_assum on those (with others as hard).
    """
    use_assumption_literals = True
    for con in soft_constraints:
        # see if solver supports reification of 'con'
        print(con)
        try:
            m = Model([BoolVar().implies(con)])
            SolverLookup.lookup(solver)(m).solve()
        except:
            # it did not
            use_assumption_literals = False

    if use_assumption_literals:
        return ocus_assum(soft_constraints, soft_weights, hard_constraints=hard_constraints, verbose=verbose)
    else:
        return ocus_pure(soft_constraints, soft_weights, hard_constraints=hard_constraints, verbose=verbose)

def ocus_pure(soft_constraints, soft_weights, hard_constraints=[], solver='ortools', verbose=False):
    # small optimisation: pre-flatten all constraints once
    # so it needs not be done over-and-over in solving
    hard = flatten_constraint(hard_constraints) # batch flatten
    soft = [flatten_constraint(c) for c in soft_constraints]

    ## Mip model
    if Model(hard+soft).solve():
        if verbose:
            print("Unexpectedly, the model is SAT")
        return []

    hs_vars = boolvar(shape=len(soft_constraints), name="hs_vars")

    hs_mip_model = Model(
        # Objective: min sum(x_l * w_l)
        minimize=sum(var * soft_weights[id] for id, var in enumerate(hs_vars))
    )

    # instantiate hitting set solver
    hittingset_solver = SolverLookup.lookup(solver)(hs_mip_model)

    while(True):
        hittingset_solver.solve()

        # Get hitting set
        hs_soft = [soft[id] for id, hs_var in enumerate(hs_vars) if hs_var.value() == 1]

        if not Model(hard+hs_soft).solve():
            if verbose > 1:
                print("\n\t ===> OCUS =", hs_soft)

            return [soft_constraints[id] for id, hs_var in enumerate(hs_vars) if hs_var.value() == 1]

        # compute complement of model in formula F
        C = hs_vars[hs_vars.value() != 1]

        # Add complement as a new set to hit: sum x[j] * hij >= 1
        hittingset_solver += (sum(C) >= 1)

        if verbose > 1:
            print("\t Complement =", C)

def ocus_assum(soft_constraints, soft_weights, hard_constraints=[], solver='ortools', verbose=False):
    # init with hard constraints
    assum_model = Model(hard_constraints)

    # make assumption indicators, add reified constraints
    ind = BoolVar(shape=len(soft_constraints), name="ind")
    for i,bv in enumerate(ind):
        assum_model += [bv.implies(soft_constraints[i])]
    # to map indicator variable back to soft_constraints
    indmap = dict((v,i) for (i,v) in enumerate(ind))

    assum_solver = SolverLookup.lookup(solver)(assum_model)

    if assum_solver.solve(assumptions=ind):
        if verbose:
            print("Unexpectedly, the model is SAT")
        return []

    ## ----------------- MODEL ------------------
    hs_mip_model = Model(
        # Objective: min sum(x_l * w_l)
        minimize=sum(var * soft_weights[id] for id, var in enumerate(ind))
    )

    # instantiate hitting set solver
    hittingset_solver = SolverLookup.lookup(solver)(hs_mip_model)

    while(True):
        hittingset_solver.solve()

        # Get hitting set
        hs = ind[ind.value() == 1]

        if not assum_solver.solve(assumptions=hs):
            if verbose > 1:
                print("\n\t ===> OCUS =", hs)

            return [soft_constraints[indmap[v]] for v in hs]

        # compute complement of model in formula F
        C = set(v for v in ind if not v.value())

        # Add complement as a new set to hit: sum x[j] * hij >= 1
        hittingset_solver += (sum(C) >= 1)

        if verbose > 1:
            print("\t Complement =", C)

if __name__ == '__main__':
    main(verbose=2)
