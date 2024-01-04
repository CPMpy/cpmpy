"""
Deletion-based Minimum Unsatisfiable Subset (MUS) algorithm.

Loosely based on PySat's MUSX:
https://github.com/pysathq/pysat/blob/master/examples/musx.py

"""
import numpy as np
import cpmpy as cp
from cpmpy.transformations.get_variables import get_variables
from cpmpy.transformations.normalize import toplevel_list

from .utils import make_assump_model


def mus(soft, hard=[], solver="ortools"):
    """
        A CP deletion-based MUS algorithm using assumption variables
        and unsat core extraction

        For solvers that support s.solve(assumptions=...) and s.get_core()

        Each constraint is an arbitrary CPMpy expression, so it can
        also be sublists of constraints (e.g. constraint groups),
        contain aribtrary nested expressions, global constraints, etc.

        Will extract an unsat core and then shrink the core further
        by repeatedly ommitting one assumption variable.

        :param: soft: soft constraints, list of expressions
        :param: hard: hard constraints, optional, list of expressions
        :param: solver: name of a solver, see SolverLookup.solvernames()
            "z3" and "gurobi" are incremental, "ortools" restarts the solver
    """
    # make assumption (indicator) variables and soft-constrained model
    (m, soft, assump) = make_assump_model(soft, hard=hard)
    s = cp.SolverLookup.get(solver, m)

    # create dictionary from assump to soft
    dmap = dict(zip(assump, soft))

    # setting all assump vars to true should be UNSAT
    assert not s.solve(assumptions=assump), "MUS: model must be UNSAT"
    core = s.get_core()  # start from solver's UNSAT core

    # order so that constraints with many variables are tried and removed first
    core = sorted(core, key=lambda c: -len(get_variables(dmap[c])))

    # deletion-based MUS
    mus = []
    for i in range(len(core)):
        subassump = mus + core[i + 1:]  # all but the 'i'th constraint

        if s.solve(assumptions=subassump):
            # removing 'i' makes the problem SAT, must keep for UNSAT
            mus.append(core[i])
        # else: still UNSAT so don't need this candidate, not in mus

    return [dmap[avar] for avar in mus]


def quickxplain(soft, hard=[], solver="ortools"):
    """
        Find a preferred minimal unsatisfiable subset of constraints, based on the ordering of constraints.

        A total order is imposed on the constraints using the ordering of `soft`.
        Constraints with lower index are preferred over ones with higher index

        Assumption-based implementation for solvers that support s.solve(assumptions=...) and s.get_core()
        More naive version available as `quickxplain_naive` to use with other solvers.

        CPMpy implementation of the QuickXplain algorithm by Junker:
            Junker, Ulrich. "Preferred explanations and relaxations for over-constrained problems." AAAI-2004. 2004.
            https://cdn.aaai.org/AAAI/2004/AAAI04-027.pdf
    """

    model, soft, assump = make_assump_model(soft, hard)
    s = cp.SolverLookup.get(solver, model)

    assert s.solve(assumptions=assump) is False, "The model should be UNSAT!"
    dmap = dict(zip(assump, soft))

    # the recursive call
    def do_recursion(soft, hard, delta):

        if len(delta) != 0 and s.solve(assumptions=hard) is False:
            # conflict is in hard constraints, no need to recurse
            return []

        if len(soft) == 1:
            # conflict is not in hard constraints, but only 1 soft constraint
            return list(soft)  # base case of recursion

        split = len(soft) // 2  # determine split point
        more_preferred, less_preferred = soft[:split], soft[split:]  # split constraints into two sets

        # treat more preferred part as hard and find extra constants from less preferred
        delta2 = do_recursion(less_preferred, hard + more_preferred, more_preferred)
        # find which preferred constraints exactly
        delta1 = do_recursion(more_preferred, hard + delta2, delta2)
        return delta1 + delta2

    # optimization: find max index of solver core
    solver_core = frozenset(s.get_core())
    max_idx = max(i for i, a in enumerate(assump) if a in solver_core)

    core = do_recursion(list(assump)[:max_idx + 1], [], [])
    return [dmap[a] for a in core]


## Naive, non-assumption based versions of MUS-algos above
def mus_naive(soft, hard=[], solver="ortools"):
    """
        A naive pure CP deletion-based MUS algorithm

        Will repeatedly solve the problem from scratch with one less constraint
        For anything but tiny sets of constraints, this will be terribly slow.

        Best to only use for testing on solvers that do not support assumptions.
        For others, use `mus()`

        :param: soft: soft constraints, list of expressions
        :param: hard: hard constraints, optional, list of expressions
        :param: solver: name of a solver, see SolverLookup.solvernames()
    """
    # ensure toplevel list
    soft = toplevel_list(soft, merge_and=False)

    m = cp.Model(hard + soft)
    assert not m.solve(solver=solver), "MUS: model must be UNSAT"

    mus = []
    # order so that constraints with many variables are tried and removed first
    core = sorted(soft, key=lambda c: -len(get_variables(c)))
    for i in range(len(core)):
        subcore = mus + core[i + 1:]  # check if all but 'i' makes core SAT

        if cp.Model(hard + subcore).solve(solver=solver):
            # removing it makes it SAT, must keep for UNSAT
            mus.append(core[i])
        # else: still UNSAT so don't need this candidate

    return mus


def quickxplain_naive(soft, hard=[], solver="ortools"):
    """
        Find a preferred minimal unsatisfiable subset of constraints, based on the ordering of constraints.

        A total order is imposed on the constraints using the ordering of `soft`.
        Constraints with lower index are preferred over ones with higher index

        Naive implementation, re-solving the model from scratch.
        Can be slower depending on the number of global constraints used and solver support for reified constraints.

        CPMpy implementation of the QuickXplain algorithm by Junker:
            Junker, Ulrich. "Preferred explanations and relaxations for over-constrained problems." AAAI-2004. 2004.
            https://cdn.aaai.org/AAAI/2004/AAAI04-027.pdf
    """

    soft = toplevel_list(soft, merge_and=False)
    assert cp.Model(hard + soft).solve(solver) is False, "The model should be UNSAT!"

    # the recursive call
    def do_recursion(soft, hard, delta):

        m = cp.Model(hard)
        if len(delta) != 0 and m.solve(solver) is False:
            # conflict is in hard constraints, no need to recurse
            return []

        if len(soft) == 1:
            # conflict is not in hard constraints, but only 1 soft constraint
            return list(soft)  # base case of recursion

        split = len(soft) // 2  # determine split point
        more_preferred, less_preferred = soft[:split], soft[split:]  # split constraints into two sets

        # treat more preferred part as hard and find extra constants from less preferred
        delta2 = do_recursion(less_preferred, hard + more_preferred, more_preferred)
        # find which preferred constraints exactly
        delta1 = do_recursion(more_preferred, hard + delta2, delta2)
        return delta1 + delta2

    core = do_recursion(soft, hard, [])
    return core
