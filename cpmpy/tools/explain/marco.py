"""
 Re-implementation of MUS-enumeration using MARCO.
"""

import cpmpy as cp
from cpmpy.transformations.get_variables import get_variables

from .utils import make_assump_model



def marco(soft, hard=[], solver="ortools", map_solver="ortools", return_mus=True, return_mcs=True, do_solution_hint=True):
    """
        Enumerating minimal unsatisfiable subsets (MUSes) and minimal correction sets (MCSes)
         of unsatisfiable constraints.
        Iteratively generates a subset of constraints (the seed) and checks whether it is SAT.
        When the seed is SAT, it is grown to an MSS to derive an MCS.
        This MCS is then added to the Map solver as a set to hit
        In case the seed is UNSAT, the seed is shrunk to a real MUS and returned.
        The Map-solver is instructed to not generated any superset of this MUS as next seeds

        Based on:
            Liffiton, Mark H., et al. "Fast, flexible MUS enumeration." Constraints 21 (2016): 223-250.

        :param: solver: the SAT-solver to use, must support assumptions, (e.g, ortools, exact, z3 or pysat)
        :param: map_solver: the hitting-set (MAP) solver to use, ideally incremental such as gurobi, pysat or exact
        :param: return_mus: whether the algorithm should return MUSes
        :param: return_mcs: whether the algorithm should return MCSes
        :param: do_solution_hint: when true, will favor large seeds generated by the map-solver, and hence more likely
                                     to return MUSes. Especially useful when `return_mus=True`.

    """

    assert hasattr(cp.SolverLookup.get(solver), "get_core"), "MARCO requires a solver that supports assumption variables"

    model, soft, assump = make_assump_model(soft, hard)
    dmap = dict(zip(assump, soft))
    s = cp.SolverLookup.get(solver, model)

    # map solver for computing hitting sets
    map_solver = cp.SolverLookup.get(map_solver)
    do_solution_hint = do_solution_hint and hasattr(map_solver, 'solution_hint')  # solver may not support solution hinting...

    map_solver += cp.any(assump)
    if do_solution_hint:
        hint = [1]*len(assump)
        map_solver.solution_hint(assump, hint) # we want large subsets, more likely to be a MUS

    deletion_order = {a : -len(get_variables(dmap[a])) for a in assump} # avoid recomputing

    while map_solver.solve():

        seed = [a for a in assump if a.value()]

        if s.solve(assumptions=seed) is True:
            # SAT, grow, to full MSS
            # Assumptions encode indicator constraints a -> c, find all true assumptions
            #    and those that could just as well be made true given the current solution
            mss = [a for a,c in zip(assump, soft) if a.value() or c.value()]
            for to_add in set(assump) - set(mss):
                if s.solve(assumptions=mss + [to_add]) is True:
                    mss.append(to_add)
            mcs = [a for a in assump if a not in frozenset(mss)] # take complement
            map_solver += cp.any(mcs) # block in map solver

            if return_mcs:
                yield "MCS", [dmap[a] for a in mcs]


        else: # UNSAT, shrink to MUS, re-use MUSX
            core = set(s.get_core())
            for c in sorted(core, key=deletion_order.get):
                if c not in core: # already removed
                    continue
                core.remove(c)
                if s.solve(assumptions=list(core)):
                    core.add(c)
                else: # UNSAT, shrink to new solver core (clause set refinement)
                    core = set(s.get_core())

            map_solver += ~cp.all(core) # block in map solver

            if return_mus:
                yield "MUS", [dmap[a] for a in core]


        # ensure solution hint is still active
        if do_solution_hint:
            map_solver.solution_hint(assump, hint)


