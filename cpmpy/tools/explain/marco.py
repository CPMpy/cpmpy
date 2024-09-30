"""
 Re-implementation of MUS-enumeration using MARCO.
"""

import cpmpy as cp
from cpmpy.transformations.get_variables import get_variables

from .utils import make_assump_model



def marco(soft, hard=[], solver="ortools", map_solver="ortools", return_mus=True, return_mcs=True):
    """
        Enumerates all MUSes in the unsatisfiable problem
        Iteratively generates a subset of constraints (the seed) and checks whether it is SAT.
        When the seed is SAT, it is grown to an MSS to derive an MCS.
        This MCS is then added to the Map solver as a set to hit
        In case the seed is UNSAT, the seed is shrunk to a real MUS and returned.
        The Map-solver is instructed to not generated any superset of this MUS as next seeds

        Based on:
            Liffiton, Mark H., et al. "Fast, flexible MUS enumeration." Constraints 21 (2016): 223-250.
    """

    model, soft, assump = make_assump_model(soft, hard)
    dmap = dict(zip(assump, soft))
    s = cp.SolverLookup.get(solver, model)

    map_solver = cp.SolverLookup.get(map_solver)
    map_solver += cp.any(assump)
    map_solver.solution_hint(assump, [1]*len(assump)) # we want large subsets, more likely to be a MUS

    while map_solver.solve():

        seed = [a for a in assump if a.value()]

        if s.solve(assumptions=seed) is True: # SAT subset, find MCSes subset(s)

            if return_mcs:
                mss = [a for a,c in zip(assump, soft) if a.value() or c.value()]
                # grow to full MSS
                for to_add in set(assump) - set(mss):
                    if s.solve(assumptions=mss + [to_add]) is True:
                        mss.append(to_add)
                mcs = [a for a in assump if a not in set(mss)]
                map_solver += cp.any(mcs)
                yield "MCS", [dmap[a] for a in mcs]

            else:
                # find more MCSes, disjoint from this one, similar to "optimal_mus" in mus.py
                # can only be done when MCSes do not have to be returned as there is no guarantee
                # the MCSes encountered during enumeration are "new" MCSes
                sat_subset = [a for a,c in zip(assump, soft) if not (a.value() or c.value())]
                map_solver += cp.any(sat_subset)
                while s.solve(assumptions=sat_subset) is True:
                    mss = [a for a, c in zip(assump, soft) if a.value() or c.value()]
                    new_mcs = [a for a in assump if a not in set(mss)]  # just take the complement
                    map_solver += cp.any(new_mcs)
                    sat_subset += new_mcs # extend sat subset with this MCS

        else: # UNSAT, shrink to MUS, re-use MUSX
            core = set(s.get_core())
            for c in sorted(core, key=lambda a : len(get_variables(dmap[a]))):
                if c not in core: # already removed
                    continue
                core.remove(c)
                if s.solve(assumptions=list(core)):
                    core.add(c)
                else: # UNSAT, shrink to new solver core (clause set refinement)
                    core = set(s.get_core())

            if return_mus:
                yield "MUS", [dmap[a] for a in core]

            # block in map solver
            map_solver += ~cp.all(core)
