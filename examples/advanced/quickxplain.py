"""
    CPMpy implementation of the QuickXplain algorithm by Junker.
        Junker, Ulrich. "Preferred explanations and relaxations for over-constrained problems." AAAI-2004. 2004.
        https://cdn.aaai.org/AAAI/2004/AAAI04-027.pdf
"""

import cpmpy as cp
from cpmpy.transformations.get_variables import get_variables
from cpmpy.transformations.normalize import toplevel_list


def quickXplain(soft, hard=[], solver="ortools"):
    """
        Find a preferred minimal unsatisfiable subset of constraints
        A partial order is imposed on the constraints using the order of `soft`.
        Constraints with lower index are preferred over ones with higher index

        :param: soft: soft constraints, list of expressions
        :param: hard: hard constraints which cannot be relaxed, optional, list of expressions
        :param: solver: name of a solver, see SolverLookup.solvernames()
            "z3", "pysat" and "gurobi" are incremental, "ortools" restarts the solver
    """

    soft = toplevel_list(soft, merge_and=False)
    assump = cp.boolvar(shape=len(soft))
    solver = cp.SolverLookup.get(solver, cp.Model(hard))
    solver += assump.implies(soft)

    assert solver.solve(assumptions=assump) is False, "The model should be UNSAT!"

    dmap = dict(zip(assump, soft))
    core = recurse_explain(list(assump), [], [], solver=solver)
    return [dmap[a] for a in core]


def recurse_explain(soft, hard, delta, solver):
    if len(delta) != 0 and solver.solve(assumptions=hard) is False:
        # conflict is in hard constraints, no need to recurse
        return []

    if len(soft) == 1:
        # conflict is not in hard constraints, but only 1 soft constraint
        return list(soft)  # base case of recursion

    split = len(soft) // 2  # determine split point
    more_preferred, less_preferred = soft[:split], soft[split:]  # split constraints into two sets

    # treat more preferred part as hard and find extra constants from less preferred
    delta2 = recurse_explain(less_preferred, hard + more_preferred, more_preferred, solver=solver)
    # find which preferred constraints exactly
    delta1 = recurse_explain(more_preferred, hard + delta2, delta2, solver=solver)
    return delta1 + delta2


if __name__ == "__main__":
    # example of quickXplain paper
    options = {"roof racks": 500,
               "CD-player": 500,
               "one additional seat": 800,
               "metal color": 500,
               "special luxury version": 2600}

    preferences = [cp.boolvar(name=name) for name in options.keys()]
    p1, p2, p3, p4, p5 = preferences

    hard = cp.sum(var * options[var.name] for var in preferences) <= 3000
    core1 = quickXplain([p1, p2, p3, p4, p5], hard)
    print("One cannot combine these options:", core1)

    core2 = quickXplain([p3, p1, p2, p5, p4], hard)
    print("One cannot combine these options:", core2)