############################################
# Copyright (c) 2016 Microsoft Corporation
# 
# Basic core and correction set enumeration.
#
# Author: Nikolaj Bjorner (nbjorner)
############################################

"""
Based on Z3's 'Basic core and correction set enumeration' implementation of Marco,
by Nikolaj Bjorner (nbjorner)
https://github.com/Z3Prover/z3/blob/master/examples/python/mus/marco.py

Adapted to CPMpy by Tias Guns.

(explanation from Z3)
Enumeration of Minimal Unsatisfiable Cores and Maximal Satisfying Subsets
This tutorial illustrates how to use (CPMpy) for extracting all minimal unsatisfiable
cores together with all maximal satisfying subsets. 

Origin
The algorithm that we describe next represents the essence of the core extraction
procedure by Liffiton and Malik and independently by Previti and Marques-Silva: 
 Enumerating Infeasibility: Finding Multiple MUSes Quickly
 Mark H. Liffiton and Ammar Malik
 in Proc. 10th International Conference on Integration of Artificial Intelligence (AI)
 and Operations Research (OR) techniques in Constraint Programming (CPAIOR-2013), 160-175, May 2013. 

Partial MUS Enumeration
 Alessandro Previti, Joao Marques-Silva in Proc. AAAI-2013 July 2013 

It illustrates the following features of CPMpy's Python-based direct access to or-tools:
   1. Using assumptions to track unsatisfiable cores. 
   2. Using multiple models/solvers and passing constraints between them. 

Idea of the Algorithm
The main idea of the algorithm is to maintain two logical contexts and exchange information
between them:

    1. The MapSolver is used to enumerate sets of clauses that are not already
       supersets of an existing unsatisfiable core and not already a subset of a maximal satisfying
       assignment. The MapSolver uses one unique atomic predicate per soft clause, so it enumerates
       sets of atomic predicates. For each minimal unsatisfiable core, say, represented by predicates
       p1, p2, p5, the MapSolver contains the clause  !p1 | !p2 | !p5. For each maximal satisfiable
       subset, say, represented by predicats p2, p3, p5, the MapSolver contains a clause corresponding
       to the disjunction of all literals not in the maximal satisfiable subset, p1 | p4 | p6. 
    2. The SubsetSolver contains a set of soft clauses (clauses with the unique indicator atom occurring negated).
       The MapSolver feeds it a set of clauses (the indicator atoms). Recall that these are not already a superset
       of an existing minimal unsatisfiable core, or a subset of a maximal satisfying assignment. If asserting
       these atoms makes the SubsetSolver context infeasible, then it finds a minimal unsatisfiable subset
       corresponding to these atoms. If asserting the atoms is consistent with the SubsetSolver, then it
       extends this set of atoms maximally to a satisfying set. 
"""

import sys
from cpmpy import *
from cpmpy.transformations.get_variables import get_variables
from cpmpy.solvers.ortools import CPM_ortools

def main():
    x = intvar(-9, 9, name="x")
    y = intvar(-9, 9, name="y")
    m = Model(
        x < 0, 
        x < 1,
        x > 2,
        (x + y > 0) | (y < 0),
        (y >= 0) | (x >= 0),
        (y < 0) | (x < 0),
        (y > 0) | (x < 0),
    )
    assert (m.solve() is False)

    print(m)
    print("\nStart MUS/MSS enumeration:")
    # Warning, all constraints must support reification...
    for kind, exprs in do_marco(m):
        print(f"{kind} {exprs}")
    
    # Use to debug your model as follows:
    #for kind, exprs in do_marco(m):
    #    if kind == 'MUS':
    #        print("Found Minimal Unsatisfiable Subset:", exprs)
    #        break


def do_marco(mdl): #csolver, map):
    """
        Basic MUS/MCS enumeration, as a simple example.
        
        Warning: all constraints in 'mdl' must support reification!
        Otherwise, you will get an "Or-tools says: invalid" error.
    """
    sub_solver = SubsetSolver(mdl.constraints)
    map_solver = MapSolver(len(mdl.constraints))

    while True:
        seed = map_solver.next_seed()
        if seed is None:
            # all MUS/MSS enumerated
            return

        if sub_solver.check_subset(seed):
            MSS = sub_solver.grow(seed)
            yield ("MSS", [mdl.constraints[i] for i in MSS])
            map_solver.block_down(MSS)
        else:
            seed = sub_solver.seed_from_core()
            MUS = sub_solver.shrink(seed)
            yield ("MUS", [mdl.constraints[i] for i in MUS])
            map_solver.block_up(MUS)


class SubsetSolver:
    def __init__(self, constraints, warmstart=True):
        n = len(constraints)
        self.all_n = set(range(n))  # used for complement

        # intialise indicators
        self.indicators = BoolVar(shape=n)
        self.idcache = dict((v,i) for (i,v) in enumerate(self.indicators))

        # make reified model
        mdl_reif = Model([ self.indicators[i].implies(con) for i,con in enumerate(constraints) ])
        self.solver = CPM_ortools(mdl_reif)

        if warmstart:
            self.warmstart = warmstart
            # for warmstarting from a previous solution
            self.user_vars = get_variables(constraints)
            self.user_vars_sol = None

    def check_subset(self, seed):
        assump = [self.indicators[i] for i in seed]
        if self.warmstart and self.user_vars_sol is not None:
            # or-tools is not incremental,
            # but we can warmstart with previous solution
            self.solver.solution_hint(self.user_vars, self.user_vars_sol)

        ret = self.solver.solve(assumptions=assump)
        if self.warmstart and ret is not False:
            # store solution for warm start
            self.user_vars_sol = [v.value() for v in self.user_vars]

        return ret

    def seed_from_core(self):
        core = self.solver.get_core()
        return set(self.idcache[v] for v in core)

    def shrink(self, seed):
        current = set(seed) # will change during loop
        for i in seed:
            if i not in current:
                continue
            current.remove(i)
            if not self.check_subset(current):
                # if UNSAT, shrink to its core
                current = self.seed_from_core()
            else:
                # without 'i' its SAT, so add back
                current.add(i)
        return current

    def grow(self, seed):
        current = seed
        for i in (self.all_n).difference(seed): # complement
            current.append(i)
            if not self.check_subset(current):
                # if UNSAT, do not add in grow
                current.pop()
        return current


class MapSolver:
    def __init__(self, n):
        """Initialization.
                Args:
               n: The number of constraints to map.
        """
        self.all_n = set(range(n))  # used for complement

        self.indicators = BoolVar(shape=n)
        # default to true for first next_seed(), "high bias"
        for v in self.indicators:
            v._value = True

        # empty model
        self.solver = CPM_ortools(Model([]))

    def next_seed(self):
        """Get the seed from the current model, if there is one.
               Returns:
               A seed as an array of 0-based constraint indexes.
        """
        if self.solver.solve() is False:
            return None
        return [i for i,v in enumerate(self.indicators) if v.value()]

    def block_down(self, frompoint):
        """Block down from a given set."""
        complement = (self.all_n).difference(frompoint)
        self.solver += any(self.indicators[i] for i in complement)

    def block_up(self, frompoint):
        """Block up from a given set."""
        self.solver += any(~self.indicators[i] for i in frompoint)
     

if __name__ == '__main__':
    main()
