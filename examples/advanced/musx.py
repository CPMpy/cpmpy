"""
Deletion-based Minimum Unsatisfiable Subset (MUS) algorithm.

Loosely based on PySat's MUSX:
https://github.com/pysathq/pysat/blob/master/examples/musx.py

"""

import sys
import copy
from cpmpy import *
from cpmpy.solvers.ortools import CPM_ortools
from cpmpy.transformations.get_variables import get_variables
from cpmpy.transformations.flatten_model import flatten_constraint

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
        AllDifferent(x,y) # invalid for musx_assum
    )
    assert (m.solve() is False)

    print(m)
    print("\nStart MUS search:")
    mus = musx(m.constraints, [], verbose=True)
    print("MUS:", mus)


def musx(soft_constraints, hard_constraints=[], verbose=False):
    """
        Deletion-based MUS for CP

        Each constraint is an arbitrary CPMpy expression, so it can
        also be sublists of constraints (e.g. constraint groups),
        contain aribtrary nested expressions, global constraints, etc.

        Will first check which soft contraints support reification,
        and use musx_assum on those (with others as hard).
        Then use musx_pure on the remaining (with current mus as hard)
    """
    soft_assum = []
    soft_pure = []
    for con in soft_constraints:
        # see if solver supports reification of 'con'
        try:
            m = Model([BoolVar().implies(con)])
            CPM_ortools(m).solve()
            # it did
            soft_assum.append(con)
        except:
            # it did not
            soft_pure.append(con)

    # find MUS of soft_assum with soft_pure as hard
    mus_assum = musx_assum(soft_assum, hard_constraints=hard_constraints+soft_pure, verbose=verbose)

    # find MUS of soft_pure with mus_assum as hard
    mus_pure = musx_pure(soft_pure, hard_constraints=hard_constraints+mus_assum, verbose=verbose)

    return mus_assum+mus_pure


def musx_pure(soft_constraints, hard_constraints=[], verbose=False):
    """
        A naive pure-CP deletion-based MUS algorithm

        Will repeatedly solve the problem with one less constraint
        For normally-sized models, this will be terribly slow.

        Best is to use this only on constraints that do not support
        reification/assumption variables (e.g. some global constraints
        with expensive decompositions).
        For those constraints that do support reification, see musx_assum()
    """
    if len(soft_constraints) == 0:
        return []

    # small optimisation:
    # order so that constraints with many variables are tried first
    # this will favor MUS with few variables per constraint,
    # and will remove large constraints earlier which may speed it up
    # TODO: count nr of subexpressions? (generalisation of nr of vars)
    soft_constraints = sorted(soft_constraints, key=lambda c: -len(get_variables(c)))

    # small optimisation: pre-flatten all constraints once
    # so it needs not be done over-and-over in solving
    hard = flatten_constraint(hard_constraints) # batch flatten
    soft = [flatten_constraint(c) for c in soft_constraints]

    if Model(hard+soft).solve():
        if verbose:
            print("Unexpectedly, the model is SAT")
        return []

    mus_idx = [] # index into 'soft_constraints' that belong to the MUS

    # init solver with hard constraints
    #s_base = CPM_ortools(Model(hard))
    m_base = Model(hard)
    for i in range(len(soft_constraints)):
        #s_without_i = copy.deepcopy(s_base) # deep copy solver state
        # add all other remaining (flattened) constraints
        s_without_i = CPM_ortools(m_base)
        s_without_i += soft[i+1:] 

        if s_without_i.solve():
            # with all but 'i' it is SAT, so 'i' belongs to the MUS
            if verbose:
                print("\tSAT so in MUS:", soft_constraints[i])
            mus_idx.append(i)
            m_base += [soft[i]]
        else:
            # still UNSAT, 'i' does not belong to the MUS
            if verbose:
                print("\tUNSAT so not in MUS:", soft_constraints[i])

    # return the list of original (non-flattened) constraints
    return [soft_constraints[i] for i in mus_idx]


def musx_assum(soft_constraints, hard_constraints=[], verbose=False):
    """
        An CP deletion-based MUS algorithm using assumption variables
        and unsat core extraction

        Will extract an unsat core and then shrink the core further
        by repeatedly ommitting one assumption variable.

        Each constraint is an arbitrary CPMpy expression, so it can
        also be sublists of constraints (e.g. constraint groups),
        contain aribtrary nested expressions, global constraints, etc.

        This approach assumes that each soft_constraint supports
        reification, that is that BoolVar().implies(constraint)
        is supported by the solver or can be efficiently decomposed
        (which may not be the case for certain global constraints)
    """
    if len(soft_constraints) == 0:
        return []

    # init with hard constraints
    assum_model = Model(hard_constraints)

    # make assumption indicators, add reified constraints
    ind = BoolVar(shape=len(soft_constraints), name="ind")
    for i,bv in enumerate(ind):
        assum_model += [bv.implies(soft_constraints[i])]
    # to map indicator variable back to soft_constraints
    indmap = dict((v,i) for (i,v) in enumerate(ind))

    # make solver once, check that it is unsat and start from core
    assum_solver = CPM_ortools(assum_model)
    if assum_solver.solve(assumptions=ind):
        if verbose:
            print("Unexpectedly, the model is SAT")
        return []
    else:
        # unsat core is an unsatisfiable subset
        mus_vars = assum_solver.get_core()
        if verbose:
            assert (not assum_solver.solve(assumptions=mus_vars)), "core is SAT!?"
        
    # now we shrink the unsatisfiable subset further
    i = 0 # we wil dynamically shrink mus_vars
    while i < len(mus_vars):
        # add all other remaining literals
        assum_lits = mus_vars[:i] + mus_vars[i+1:]

        if assum_solver.solve(assumptions=assum_lits):
            # with all but 'i' it is SAT, so 'i' belongs to the MUS
            if verbose:
                print("\tSAT so in MUS:", soft_constraints[indmap[mus_vars[i]]])
            i += 1
        else:
            # still UNSAT, 'i' does not belong to the MUS
            if verbose:
                print("\tUNSAT so not in MUS:", soft_constraints[indmap[mus_vars[i]]])
            # overwrite current 'i' and continue
            # could do get_core but then have to check that mus_vars[:i] match
            mus_vars = assum_lits


    # return the list of original (non-flattened) constraints
    return [soft_constraints[indmap[v]] for v in mus_vars]


if __name__ == '__main__':
    main()
