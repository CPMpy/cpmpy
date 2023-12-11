from cpmpy.tools.explain.mss import *
from cpmpy.transformations.normalize import toplevel_list


def mcs(soft, hard=[], solver="ortools"):
    """
        Compute Minimal Correction Subset of unsatisfiable model.
        Remvoving these contraints will result in a satisfiable model.
        Computes a subset of constraints which minimizes the total number of constraints to be removed
    """
    return mcs_opt(soft, hard, 1, solver)


def mcs_opt(soft, hard, weights=1, solver="ortools"):
    """
        Compute Minimal Correction Subset of unsatisfiable model.
        Constraints can be weighted using the `weights` parameter.
        Computes a subset of constraints which minimizes the sum of all weights of constraints.
    """
    soft2 = toplevel_list(soft, merge_and=False)
    mymss = mss_opt(soft2, hard, weights, solver=solver)
    return list(set(soft2) - set(mymss))


def mcs_grow(soft, hard, solver="ortools"):
    """
        Computes correction subset without requirement of optimization support
        Relies on assumptions so incremental solvers are adviced.
        Can be faster in some cases compared to optimal correction subset
    """
    soft2 = toplevel_list(soft, merge_and=False)
    mymss = mss_grow(soft2, hard, solver)
    return list(set(soft2) - set(mymss))


def mcs_grow_naive(soft, hard, solver="ortools"):
    """
        Compute Minimal Correction Subset of unsatsifiable model.
        Computes a subset-minimal set of constraints by greedily removing contraints.
        Can be used when solver does not support assumptions
        No guarantees on optimality, but can be faster in some cases
    """
    soft2 = toplevel_list(soft, merge_and=False)
    mymss = mss_grow_naive(soft2, hard, solver)
    return list(set(soft2) - set(mymss))
