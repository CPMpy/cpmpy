import cpmpy as cp
from cpmpy.expressions.utils import is_any_list
from cpmpy.transformations.normalize import toplevel_list

from .utils import make_assump_model

# Maximum Satisfiable Subset
# assumes the solver supports 'maximize', if not... revert to 'mss_grow'
def mss(soft, hard=[], solver="ortools"):
    """
        Compute Maximal Satisfiable Subset of unsatisfiable model.
        Computes a subset of constraints which maximises the total number of constraints
    """
    return mss_opt(soft, hard, 1, solver)


def mss_opt(soft, hard=[], weights=1, solver="ortools"):
    """
        Compute Maximal Satisfiable Subset of unsatisfiable model.
        Constraints can be weighted using the `weights` parameter.
        Computes a subset of constraints which maximizes the sum of all weights on constraints
    """
    # make assumption (indicator) variables and soft-constrained model
    (m, soft, assump) = make_assump_model(soft, hard=hard)
    s = cp.SolverLookup.get(solver, m)

    # maximize weight of indicator vars
    if is_any_list(weights):
        assert len(weights) == len(assump)

    s.maximize(cp.sum(weights * assump))
    assert s.solve()

    return [c for avar, c in zip(assump, soft) if avar.value()]


def mss_grow(soft, hard=[], solver="ortools"):
    """
        Compute Maximal Satisfiable Subset of unsatsifiable model.
        Computes a subset-maximal set of constraints by greedily adding contraints.
        Relies on solving under assumptions, so using an incremental solver is adviced
        No guarantees on optimality, but can be faster in some cases
    """

    (m, soft, assump) = make_assump_model(soft, hard=hard)
    s = cp.SolverLookup.get(solver, m)

    if hasattr(s, "solution_hint"):
        # warm start satisfiable subset
        s.solution_hint(assump, [1] * len(assump))
        assert s.solve()
        sat_subset = [a for a in assump if a.value()]
    else:
        sat_subset = []

    to_check = list(set(assump) - set(sat_subset))
    while len(to_check):
        a = to_check.pop()
        if s.solve(assumptions=sat_subset + [a]) is True:
            sat_subset.append(a)
        else:
            # UNSAT, cannot add
            pass

    sat_subset = set(sat_subset)
    return [c for avar, c in zip(assump, soft) if avar in sat_subset]


def mss_grow_naive(soft, hard=[], solver="ortools"):
    """
        Compute Maximal Satisfiable Subset of unsatsifiable model.
        Computes a subset-maximal set of constraints by greedily adding contraints.
        Can be used when solver does not support assumptions
        No guarantees on optimality, but can be faster in some cases
    """

    soft = toplevel_list(soft, merge_and=False)
    if not is_any_list(hard):
        hard = [hard]
    else:
        hard = list(hard)

    to_check = list(soft)
    sat_subset = []

    while len(to_check):
        c = to_check.pop()
        if cp.Model(sat_subset + [c] + hard).solve(solver=solver) is True:
            sat_subset += [c]
        else:
            # UNSAT, cannot add to sat subset
            pass

    return sat_subset
