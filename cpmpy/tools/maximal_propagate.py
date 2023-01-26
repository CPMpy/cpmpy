"""
    Maximal propagation of CPMpy constraints using repeated solving.
"""

from cpmpy import *
from cpmpy.transformations.get_variables import get_variables


def maximal_propagate(constraints, vars=None, solver="ortools", method="union"):
    """
        Maximal propagation algorithm for CP programs.

        Returns all values for variables for which at least one model exists.
        I.e., returns a globally consistent constraint network.

        :param: constraints: list of CPMpy constraints
        :param: vars: list of variables, optional, list of variables to propagate.
        :param: solver: name of a solver, see SolverLookup.solvernames().
                            for faster propgation, use incremental solver such as pysat, z3 or gurobi.
        :param: method: method of propagation, optional, for large domains, use 'union',
                                                    for small domains use 'intersect'
    """
    if vars is None or len(vars) == 0:
        vars = get_variables(constraints)

    if "union" in method:
        propagated = maximal_propagate_union(constraints, vars, solver)

    elif "intersect" in method:
        propagated = maximal_propagate_intersect(constraints, vars, solver)

    else:
        raise ValueError("Method to use should be one of {'union', intersect'}")

    if any(len(dom) == 0 for dom in propagated.values()):
        raise ValueError(f"Constraints {constraints} are unsatisfiable! Maximal propagation of unsat constraints not defined.")
    return propagated

def maximal_propagate_union(constraints, vars, solver="ortools"):
    """
        Maximal propagation of constraints using union of models.
        Each iteration of the algorithm requires a solution with at least one unseen variable assignment.
    """
    visisted_domain = {var: set() for var in vars}

    solver = SolverLookup.get(solver)
    solver += constraints

    while solver.solve():
        cons = False
        # visit at least one new value
        for var, values in visisted_domain.items():
            values.add(var.value())
            cons |= all(var != val for val in values)
        solver += cons
    # exhausted all possible domain values
    return visisted_domain


def maximal_propagate_intersect(constraints, vars, solver="ortools"):
    """
        Maximal propagation of constraints using intersection of models.
        Each iteration of the algorithm requires a solution with at least one unseen variable assignment.
    """
    domain_to_visit = {var : set(range(var.lb, var.ub+1)) for var in vars}

    solver = SolverLookup.get(solver)
    solver += constraints

    while solver.solve():
        cons = False
        for var, values in domain_to_visit.items():
            values.discard(var.value())
            # assign at least one variable to an unvisited value
            cons |= any(var == val for val in values)

        solver += cons

    # return values of variables reachable given the constraints
    return {var : {val for val in range(var.lb, var.ub+1) if val not in domain_to_visit[var]}
            for var in vars}