"""
 Re-implementation of MUS-enumeration using MARCO.
"""

import cpmpy as cp
from cpmpy.tools import make_assump_model



def marco(soft, hard=[], solver="ortools", map_solver="ortools"):
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

        if s.solve(assumptions=seed) is True: # SAT subset, find corr subsets
            sat_subset = seed
            new_mcs = [a for a, c in zip(assump, soft) if a.value() is False and c.value() is False]
            map_solver += cp.any(new_mcs)
            sat_subset += new_mcs
            while s.solve(assumptions=sat_subset) is True:
                new_mcs = [a for a, c in zip(assump, soft) if a.value() is False and c.value() is False]
                sat_subset += new_mcs  # extend sat subset with new MCS
                map_solver += cp.any(new_mcs)

        else: # UNSAT, shrink to MUS, re-use MUSX
            core = s.get_core()
            mus = []
            for i in range(len(core)):
                subassump = mus + core[i + 1:]  # all but the 'i'th constraint
                if s.solve(assumptions=subassump):
                    mus.append(core[i])

            yield [dmap[a] for a in mus]
            # block in map solver
            map_solver += ~cp.all(mus)
