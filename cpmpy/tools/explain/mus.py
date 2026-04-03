"""
    Re-impementation of MUS-computation techniques in CPMPy

    - Deletion-based MUS
    - QuickXplain
    - Optimal MUS
"""
import warnings
import numpy as np
import cpmpy as cp
from cpmpy.solvers.solver_interface import ExitStatus
from cpmpy.transformations.get_variables import get_variables
from cpmpy.transformations.normalize import toplevel_list

from .utils import make_assump_model, replace_cons_with_assump, OCUSException


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
        :param: solver: name of a solver, must support assumptions (e.g, "ortools", "exact", "z3" or "pysat")
    """

    assert hasattr(cp.SolverLookup.get(solver), "get_core"), f"mus requires a solver that supports assumption variables, use mus_naive with {solver} instead"

    # make assumption (indicator) variables and soft-constrained model
    (m, soft, assump) = make_assump_model(soft, hard=hard)
    s = cp.SolverLookup.get(solver, m)

    # create dictionary from assump to soft
    dmap = dict(zip(assump, soft))

    # setting all assump vars to true should be UNSAT
    assert not s.solve(assumptions=assump), "MUS: model must be UNSAT"
    core = set(s.get_core())  # start from solver's UNSAT core

    # deletion-based MUS
    # order so that constraints with many variables are tried and removed first
    for c in sorted(core, key=lambda c : -len(get_variables(dmap[c]))):
        if c not in core:
            continue # already removed
        core.remove(c)
        if s.solve(assumptions=list(core)) is True:
            core.add(c)
        else: # UNSAT, use new solver core (clause set refinement)
            core = set(s.get_core())

    return [dmap[avar] for avar in core]


def quickxplain(soft, hard=[], solver="ortools"):
    """
        Find a preferred minimal unsatisfiable subset of constraints, based on the ordering of constraints.

        A total order is imposed on the constraints using the ordering of `soft`.
        Constraints with lower index are preferred over ones with higher index

        Assumption-based implementation for solvers that support s.solve(assumptions=...) and s.get_core()
        More naive version available as `quickxplain_naive` to use with other solvers.

        :param: soft: list of soft constraints to find a preferred minimal unsatisfiable subset of
        :param: hard: list of hard constraints, will be added to the model before solving
        :param: solver: name of a solver, must support assumptions (e.g, "ortools", "exact", "z3" or "pysat")

        CPMpy implementation of the QuickXplain algorithm by Junker:
            Junker, Ulrich. "Preferred explanations and relaxations for over-constrained problems." AAAI-2004. 2004.
            https://cdn.aaai.org/AAAI/2004/AAAI04-027.pdf
    """

    assert hasattr(cp.SolverLookup.get(solver), "get_core"), f"quickxplain requires a solver that supports assumption variables, use quickxplain_naive with {solver} instead"

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

def ocus(soft, hard=[], weights=None, meta_constraint=True, solver="ortools", hs_solver="ortools", do_solution_hint=True):
    """
        Find an optimal and constrained MUS according to a linear objective function.
        By not providing a weightvector, this function will return the smallest mus.
        Works by iteratively generating correction subsets and computing optimal hitting sets to those enumerated sets.
        For better performance of the algorithm, use an incemental solver to compute the hitting sets such as Gurobi.

        Assumption-based implementation for solvers that support s.solve(assumptions=...)
        More naive version available as `optimal_mus_naive` to use with other solvers.

        :param: soft: list of soft constraints to find an optimal MUS of
        :param: hard: list of hard constraints, will be added to the model before solving
        :param: weights: list of weights for the soft constraints, will be used to compute the objective function
        :param: meta_constraint: a Boolean CPMpy expression that contains constraints in `soft` as sub-expressions.
            By not providing a meta_constraint, this function will return an optimal mus.
        :param: solver: name of a solver, must support assumptions (e.g, "ortools", "exact", "z3" or "pysat")
        :param: hs_solver: the hitting-set solver to use, ideally incremental such as "gurobi"
        :param: do_solution_hint: when true, will favor large satisfiable subsets generated by the SAT-solver

        CPMpy implementation loosely based on the "OCUS" algorithm from:

            Gamba, Emilio, Bart Bogaerts, and Tias Guns. "Efficiently explaining CSPs with unsatisfiable subset optimization."
            Journal of Artificial Intelligence Research 78 (2023): 709-746.
    """

    assert hasattr(cp.SolverLookup.get(solver), "get_core"), f"ocus requires a solver that supports assumption variables, use ocus_naive with {solver} instead"
    
    model, soft, assump = make_assump_model(soft, hard)
    dmap = dict(zip(assump, soft)) # map assumption variables to constraints

    s = cp.SolverLookup.get(solver, model)
    if do_solution_hint and hasattr(s, 'solution_hint'): # algo is constructive, so favor large subsets
        s.solution_hint(assump, [1]*len(assump))

    assert s.solve(assumptions=assump) is False

    # initialize hitting set solver
    if weights is None:
        weights = np.ones(len(assump), dtype=int)

    hs_solver = cp.SolverLookup.get(hs_solver)
    hs_solver.minimize(cp.sum(assump * np.array(weights)))

    assump_constraint = replace_cons_with_assump(meta_constraint, dict(zip(soft, assump)))
    assert set(get_variables(assump_constraint)) <= set(assump), f"soft constraints should be replaced with assumption variables by now, but got {assump_constraint}"
    hs_solver += assump_constraint

    while hs_solver.solve() is True:
        hitting_set = [a for a in assump if a.value()]
        if s.solve(assumptions=hitting_set) is False:
            break

        # else, the hitting set is SAT, now try to extend it without extra solve calls.
        # Check which other assumptions/constraints are satisfied (using c.value())
        # complement of grown subset is a correction subset
        # Assumptions encode indicator constraints a -> c, find all false assumptions
        #   that really have to be false given the current solution.
        new_corr_subset = [a for a,c in zip(assump, soft) if a.value() is False and c.value() is False]
        hs_solver += cp.sum(new_corr_subset) >= 1

        # greedily search for other corr subsets disjoint to this one
        sat_subset = list(new_corr_subset)
        while s.solve(assumptions=sat_subset) is True:
            new_corr_subset = [a for a,c in zip(assump, soft) if a.value() is False and c.value() is False]
            sat_subset += new_corr_subset # extend sat subset with new corr subset, guaranteed to be disjoint
            hs_solver += cp.sum(new_corr_subset) >= 1 # add new corr subset to hitting set solver

    if hs_solver.status().exitstatus == ExitStatus.UNSATISFIABLE:
        raise OCUSException(f"No unsatisfiable subset adhereing to constraint {meta_constraint} could be found.")

    return [dmap[a] for a in hitting_set]


def optimal_mus(soft, hard=[], weights=None, solver="ortools", hs_solver="ortools", do_solution_hint=True):
    """
        Find an optimal MUS according to a linear objective function.
    """
    return ocus(soft, hard, weights, meta_constraint=True, solver=solver, hs_solver=hs_solver, do_solution_hint=do_solution_hint)

def smus(soft, hard=[], solver="ortools", hs_solver="ortools"):
    """
        Find a smallest MUS according, equivalent to `optimal_mus` with weights=None
    """
    return optimal_mus(soft, hard=hard, weights=None, solver=solver, hs_solver=hs_solver)


## Naive, non-assumption based versions of MUS-algos above
def mus_naive(soft, hard=[], solver="ortools"):
    """
        A naive pure CP deletion-based MUS algorithm

        Will repeatedly solve the problem from scratch with one less constraint
        For anything but tiny sets of constraints, this will be terribly slow.

        Best to only use for testing on solvers that do not support assumptions.
        For others, use `mus()`

        :param soft: soft constraints, list of expressions
        :param hard: hard constraints, optional, list of expressions
        :param solver: name of a solver, see SolverLookup.solvernames()
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

def ocus_naive(soft, hard=[], weights=None, meta_constraint=True, solver="ortools", hs_solver="ortools", do_solution_hint=True):
    """
        Naive implementation of `ocus` without assumption variables and incremental solving
    """
    soft = toplevel_list(soft, merge_and=False)
    bvs = cp.boolvar(shape=(len(soft),))

    if weights is None:
        weights = np.ones(len(bvs), dtype=int)
    hs_solver = cp.SolverLookup.get(hs_solver)
    hs_solver.minimize(cp.sum(bvs * np.array(weights)))

    bv_cons = replace_cons_with_assump(meta_constraint, dict(zip(soft, bvs)))
    assert set(get_variables(bv_cons)) <= set(bvs), f"soft constraints should be replaced with boolean variables by now, but got {bv_cons}"
    hs_solver += bv_cons

    while hs_solver.solve() is True:

        hitting_set = [c for bv, c in zip(bvs, soft) if bv.value()]
        if cp.Model(hard + hitting_set).solve(solver=solver) is False:
            break

        # else, the hitting set is SAT, now try to extend it without extra solve calls.
        # Check which other assumptions/constraints are satisfied using its value() function
        #       sidenote: some vars may not be know to model and are None!
        # complement of grown subset is a correction subset
        false_constraints = [s for s in soft if s.value() is False or s.value() is None]
        corr_subset = [bv for bv,c in zip(bvs, soft) if c in frozenset(false_constraints)]
        hs_solver += cp.sum(corr_subset) >= 1

        # find more corr subsets, disjoint to this one
        sat_subset = hitting_set + false_constraints
        while cp.Model(hard + sat_subset).solve(solver=solver) is True:
            false_constraints = [s for s in soft if s.value() is False or s.value() is None]
            corr_subset = [bv for bv, c in zip(bvs, soft) if c in frozenset(false_constraints)]
            hs_solver += cp.sum(corr_subset) >= 1
            sat_subset += false_constraints

    if hs_solver.status().exitstatus == ExitStatus.UNSATISFIABLE:
        raise OCUSException("No unsatisfiable constrained subset could be found") # TODO: better exception?

    return hitting_set


    

def optimal_mus_naive(soft, hard=[], weights=None, solver="ortools", hs_solver="ortools"):
    """
        Naive implementation of `optimal_mus` without assumption variables and incremental solving
    """
    return ocus_naive(soft, hard, weights, meta_constraint=True, solver=solver, hs_solver=hs_solver)

   


