"""
Deletion-based Minimum Unsatisfiable Subset (MUS) algorithm.

Loosely based on PySat's MUSX:
https://github.com/pysathq/pysat/blob/master/examples/musx.py

"""
import numpy as np
from cpmpy import *
from cpmpy.expressions.variables import NDVarArray
from cpmpy.transformations.get_variables import get_variables
from cpmpy.transformations.normalize import toplevel_list

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
    # ensure toplevel list
    soft = toplevel_list(soft, merge_and=False)

    # order so that constraints with many variables are tried and removed first
    candidates = sorted(soft, key=lambda c: -len(get_variables(c)))

    assump = boolvar(shape=len(soft), name="assump")
    if len(soft) == 1:
        assump = NDVarArray(shape=1, dtype=object, buffer=np.array([assump]))

    m = Model(hard+[assump.implies(candidates)]) # each assumption variable implies a candidate
    s = SolverLookup.get(solver, m)
    assert not s.solve(assumptions=assump), "MUS: model must be UNSAT"

    mus = []
    core = sorted(s.get_core()) # start from solver's UNSAT core
    for i in range(len(core)):
        subassump = mus + core[i+1:]  # check if all but 'i' makes constraints SAT
        
        if s.solve(assumptions=subassump):
            # removing it makes it SAT, must keep for UNSAT
            mus.append(core[i])
        # else: still UNSAT so don't need this candidate
    
    # create dictionary from assump to candidate
    dmap = dict(zip(assump,candidates))
    return [dmap[assump] for assump in mus]
    

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
    m = Model(hard+soft)
    assert not m.solve(solver=solver), "MUS: model must be UNSAT"

    mus = []
    # order so that constraints with many variables are tried and removed first
    core = sorted(soft, key=lambda c: -len(get_variables(c)))
    for i in range(len(core)):
        subcore = mus + core[i+1:]  # check if all but 'i' makes core SAT
        
        if Model(hard+subcore).solve(solver=solver):
            # removing it makes it SAT, must keep for UNSAT
            mus.append(core[i])
        # else: still UNSAT so don't need this candidate
    
    return mus
