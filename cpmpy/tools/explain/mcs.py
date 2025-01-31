from cpmpy.tools.explain.mss import *
from cpmpy.transformations.normalize import toplevel_list


def mcs(soft, hard=[], solver="ortools", time_limit=None):
    """
        Compute Minimal Correction Subset of unsatisfiable model.
        Removing these contraints will result in a satisfiable model.
        Computes a subset of constraints which minimizes the total number of constraints to be removed

        :param time_limit: time limit in seconds (default: None)
        :return: list of constraints, or empty list if time limit is reached
    """
    result = mcs_opt(soft, hard, 1, solver, time_limit)
    return result


def mcs_opt(soft, hard, weights=1, solver="ortools", time_limit=None):
    """
        Compute Minimal Correction Subset of unsatisfiable model.
        Constraints can be weighted using the `weights` parameter.
        Computes a subset of constraints which minimizes the sum of all weights of constraints.

        :param time_limit: time limit in seconds (default: None)
        :return: list of constraints, or empty list if time limit is reached
    """
    soft2 = toplevel_list(soft, merge_and=False)
    mymss = mss_opt(soft2, hard, weights, solver=solver, time_limit=time_limit)
    # If MSS is empty due to timeout, return empty list
    if not mymss:
        return []
    return list(set(soft2) - set(mymss))


def mcs_grow(soft, hard, solver="ortools", time_limit=None):
    """
        Computes correction subset without requirement of optimization support
        Relies on assumptions so incremental solvers are adviced.
        Can be faster in some cases compared to optimal correction subset

        :param time_limit: time limit in seconds (default: None)
        :return: list of constraints, or empty list if time limit is reached
    """
    soft2 = toplevel_list(soft, merge_and=False)
    mymss = mss_grow(soft2, hard, solver, time_limit)
    # If MSS is empty due to timeout, return empty list
    if not mymss:
        return []
    return list(set(soft2) - set(mymss))


def mcs_grow_naive(soft, hard, solver="ortools", time_limit=None):
    """
        Compute Minimal Correction Subset of unsatsifiable model.
        Computes a subset-minimal set of constraints by greedily removing contraints.
        Can be used when solver does not support assumptions
        No guarantees on optimality, but can be faster in some cases

        :param time_limit: time limit in seconds (default: None)
        :return: list of constraints, or empty list if time limit is reached
    """
    soft2 = toplevel_list(soft, merge_and=False)
    mymss = mss_grow_naive(soft2, hard, solver, time_limit)
    # If MSS is empty due to timeout, return empty list
    if not mymss:
        return []
    return list(set(soft2) - set(mymss))
