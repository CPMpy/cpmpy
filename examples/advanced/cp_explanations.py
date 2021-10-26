from cpmpy import *
from cpmpy.transformations.flatten_model import flatten_constraint

import numpy as np
'''
Weighted unsatisfiable cores from given weighted unsatisfiable formula


Example Application of weighted unsatisfiable cores (cost-Optimal
unsatisfiable subsets) for explaining solutions of constraint satisfaction
problems. [1, 2, 3]

Intuition:
Uses the implicit hitting set duality between Minimum Correction Subsets (MCSes)
and Minimum Unsatisfiable Subsets (MUSes) for computing Weighted Unsatisfiable
subsets (cores).

Iteratively look for a cost-minimal hitting set on the computed MCSes so far.
- If the hitting set is SAT, it is grown to a satisfiable subset and
  the complement is added to the collection of MCSes.
- If the hitting set is UNSAT, the computed hitting set is a
  Weighted (cost-Optimal) Unsatisfiable Subset. 

References:
    [1] Gamba, E., Bogaerts, B., & Guns, T. (2021). Efficiently Explaining CSPs
    with Unsatisfiable Subset Optimization. In Proceedings of the Thirtieth
    International Joint Conference on Artificial Intelligence. Pages 1381-1388.
    https://doi.org/10.24963/ijcai.2021/191

    [2] Bogaerts, B., Gamba, E., Claes, J., & Guns, T. (2020). Step-wise explanations
    of constraint satisfaction problems. In ECAI 2020-24th European Conference on
    Artificial Intelligence, 29 August-8 September 2020, Santiago de Compostela,
    Spain, August 29-September 8, 2020-Including 10th Conference on Prestigious
    Applications of Artificial Intelligence (PAIS 2020) (Vol. 325, pp. 640-647).
    IOS Press; https://doi. org/10.3233/FAIA200149.

    [3] Bogaerts, B., Gamba, E., & Guns, T. (2021). A framework for step-wise explaining
    how to solve constraint satisfaction problems. Artificial Intelligence, 300, 103550.
'''


def main(verbose=1):
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
    print("\nStart OMUS search:")
    mus = omus(m.constraints, weights, [], verbose=verbose)
    print("OMUS:", mus)

def omus(soft_constraints, soft_weights, hard_constraints=[], solver='ortools', verbose=1):
    """
        Hitting set based weighted-MUS for CP

        Each constraint is an arbitrary CPMpy expression, so it can
        also be sublists of constraints (e.g. constraint groups),
        contain aribtrary nested expressions, global constraints, etc.

        Will first check which soft contraints support reification,
        and use omus_assum on those (with others as hard).
    """
    use_assumption_literals = True
    for con in soft_constraints:
        # see if solver supports reification of 'con'
        try:
            m = Model([BoolVar().implies(con)])
            SolverLookup.lookup(solver)(m).solve()
        except:
            # it did not
            use_assumption_literals = False

    if use_assumption_literals:
        return omus_assum(soft_constraints, soft_weights, hard_constraints=hard_constraints, verbose=verbose)
    else:
        return omus_pure(soft_constraints, soft_weights, hard_constraints=hard_constraints, verbose=verbose)

def omus_pure(soft_constraints, soft_weights, hard_constraints=[], solver='ortools', verbose=1):
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
        hs_ids = [i for i, hs_var in enumerate(hs_vars) if hs_var.value() == 1]
        hs_soft = [soft[i] for i in hs_ids]

        if not Model(hard+hs_soft).solve():
            if verbose > 1:
                cost = sum([soft_weights[i] for i in hs_ids])
                print("\t hitting set with cost", cost, "is UNSAT:", [soft_constraints[i] for i in hs_ids])

            return [soft_constraints[i] for i in hs_ids]

        if verbose > 1:
            cost = sum([soft_weights[i] for i in hs_ids])
            print("\t hitting set with cost", cost, "is SAT:", [soft_constraints[i] for i in hs_ids])

        # compute complement of model in formula F
        C = hs_vars[hs_vars.value() != 1]

        # Add complement as a new set to hit: sum x[j] * hij >= 1
        hittingset_solver += (sum(C) >= 1)


def omus_assum(soft_constraints, soft_weights, hard_constraints=[], solver='ortools', verbose=1):
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
                cost = sum([soft_weights[indmap[v]] for v in hs])
                print("\t hitting set with cost", cost, "is UNSAT:", [soft_constraints[indmap[v]] for v in hs])

            return [soft_constraints[indmap[v]] for v in hs]

        if verbose > 1:
            cost = sum([soft_weights[indmap[v]] for v in hs])
            print("\t hitting set with cost", cost, "is SAT:", [soft_constraints[indmap[v]] for v in hs])

        # compute complement of model in formula F
        C = set(v for v in ind if not v.value())

        # Add complement as a new set to hit: sum x[j] * hij >= 1
        hittingset_solver += (sum(C) >= 1)


if __name__ == '__main__':
    main(verbose=2)
