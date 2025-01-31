import cpmpy as cp
from cpmpy.expressions.utils import is_any_list
from cpmpy.transformations.normalize import toplevel_list
from cpmpy.solvers.solver_interface import ExitStatus

from .utils import make_assump_model

# Maximum Satisfiable Subset
# assumes the solver supports 'maximize', if not... revert to 'mss_grow'
def mss(soft, hard=[], solver="ortools", time_limit=None):
    """
        Compute Maximal Satisfiable Subset of unsatisfiable model.
        Computes a subset of constraints which maximises the total number of constraints

        :param time_limit: time limit in seconds (default: None)
        :return: list of constraints, or empty list if time limit is reached
    """
    return mss_opt(soft, hard, 1, solver, time_limit)


def mss_opt(soft, hard=[], weights=1, solver="ortools", time_limit=None):
    """
        Compute Maximal Satisfiable Subset of unsatisfiable model.
        Constraints can be weighted using the `weights` parameter.
        Computes a subset of constraints which maximizes the sum of all weights on constraints

        :param time_limit: time limit in seconds (default: None)
        :return: list of constraints, or empty list if time limit is reached
    """
    # make assumption (indicator) variables and soft-constrained model
    (m, soft, assump) = make_assump_model(soft, hard=hard)
    s = cp.SolverLookup.get(solver, m)

    # maximize weight of indicator vars
    if is_any_list(weights):
        assert len(weights) == len(assump)

    s.maximize(cp.sum(weights * assump))
    if s.solve(time_limit=time_limit) is False:
        return []

    return [c for avar, c in zip(assump, soft) if avar.value()]


def mss_grow(soft, hard=[], solver="ortools", time_limit=None):
    """
        Compute Maximal Satisfiable Subset of unsatsifiable model.
        Computes a subset-maximal set of constraints by greedily adding contraints.
        Relies on solving under assumptions, so using an incremental solver is adviced
        No guarantees on optimality, but can be faster in some cases

        Exploits the solution found to add more constraints at once, cfr:
        Mencía, Carlos, and Joao Marques-Silva. "Efficient relaxations of over-constrained CSPs."
        2014 IEEE 26th International Conference on Tools with Artificial Intelligence. IEEE, 2014.

        :param time_limit: time limit in seconds (default: None)
        :return: list of constraints, or empty list if time limit is reached
    """
    (m, soft, assump) = make_assump_model(soft, hard=hard)
    s = cp.SolverLookup.get(solver, m)

    if hasattr(s, "solution_hint"):
        # warm start satisfiable subset
        s.solution_hint(assump, [1] * len(assump))
        if s.solve(time_limit=time_limit) is False:
            if s.cpm_status.exitstatus == ExitStatus.UNKNOWN:
                return []
        sat_subset = [a for a in assump if a.value()]
    else:
        sat_subset = []

    to_check = set(assump) - set(sat_subset)
    while len(to_check):
        a = to_check.pop()
        if s.solve(assumptions=sat_subset + [a], time_limit=time_limit) is False:
            # UNSAT or timeout, cannot add
            return [] if s.cpm_status.exitstatus == ExitStatus.UNKNOWN else sat_subset
        sat_subset = [a for a, c in zip(assump, soft) if a.value() or c.value()]
        to_check -= set(sat_subset)

    sat_subset = set(sat_subset)
    return [c for avar, c in zip(assump, soft) if avar in sat_subset]


def mss_grow_naive(soft, hard=[], solver="ortools", time_limit=None):
    """
        Compute Maximal Satisfiable Subset of unsatsifiable model.
        Computes a subset-maximal set of constraints by greedily adding contraints.
        Can be used when solver does not support assumptions
        No guarantees on optimality, but can be faster in some cases

        :param time_limit: time limit in seconds (default: None)
        :return: list of constraints, or empty list if time limit is reached
    """
    soft = toplevel_list(soft, merge_and=False)
    if not is_any_list(hard):
        hard = [hard]
    else:
        hard = list(hard)

    to_check = list(soft)
    sat_subset = []
    s = cp.SolverLookup.get(solver)
    s += hard

    while len(to_check):
        c = to_check.pop()
        s += c
        if s.solve(time_limit=time_limit) is False:
            # UNSAT or timeout
            if s.cpm_status.exitstatus == ExitStatus.UNKNOWN:
                return []
            # UNSAT, cannot add to sat subset, reset solver to just sat subset
            s = cp.SolverLookup.get(solver)
            s += (sat_subset + hard)
        else:
            sat_subset.append(c)

    return sat_subset
