from cpmpy.tools.explain.mss import *
from cpmpy.transformations.normalize import toplevel_list


def mcs(soft, hard=[], solver="ortools"):
    """
        Compute Minimal Correction Subset of unsatisfiable model.
        Removing these contraints will result in a satisfiable model.
        Computes a subset of constraints which minimizes the total number of constraints to be removed

        :param: soft: list of soft constraints that may be part of the minimal correction subset
        :param: hard: list of hard constraints, will be added to the model before solving
        :param: solver: the SAT-solver to use, must support optimization
    """
    return mcs_opt(soft, hard, 1, solver)


def mcs_opt(soft, hard, weights=1, solver="ortools"):
    """
        Compute Minimal Correction Subset of unsatisfiable model.
        Constraints can be weighted using the `weights` parameter.
        Computes a subset of constraints which minimizes the sum of all weights of constraints.

        :param: soft: list of soft constraints that may be part of the minimal correction subset
        :param: hard: list of hard constraints, will be added to the model before solving
        :param: weights: weight of each constraint, default is 1
        :param: solver: the SAT-solver to use, must support optimization
    """
    soft2 = toplevel_list(soft, merge_and=False)
    mymss = mss_opt(soft2, hard, weights, solver=solver)
    return list(set(soft2) - set(mymss))


def mcs_grow(soft, hard, solver="ortools"):
    """
        Computes correction subset without requirement of optimization support
        Relies on assumptions so incremental solvers are adviced.
        Can be faster in some cases compared to optimal correction subset

        :param: soft: list of soft constraints that may be part of the minimal correction subset
        :param: hard: list of hard constraints, will be added to the model before solving
        :param: solver: the SAT-solver to use, must support assumptions
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

        :param: soft: list of soft constraints that may be part of the minimal correction subset
        :param: hard: list of hard constraints, will be added to the model before solving
        :param: solver: the SAT-solver to use, ideally incremental such as "gurobi", "exact"
    """
    soft2 = toplevel_list(soft, merge_and=False)
    mymss = mss_grow_naive(soft2, hard, solver)
    return list(set(soft2) - set(mymss))
