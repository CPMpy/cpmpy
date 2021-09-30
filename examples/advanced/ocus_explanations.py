from cpmpy.solvers.ortools import CPM_ortools
from cpmpy.solvers.pysat import CPM_pysat
from cpmpy.expressions.core import Operator
from cpmpy import *

import numpy as np

def print_explanations(explanations, soft_constraints, indmap):

    for expl_id, explanation in enumerate(explanations):
        print(explanation)
        print("Explanation", expl_id, "\t cost=", explanation["cost"])
        for con_id in explanation['constraints']:
            if con_id in indmap:
                print('\t Constraints=', soft_constraints[indmap[con_id]])
            else:
                print('\t Fact=', con_id)
        print('\tDerived:', explanation['derived'])


def optimal_propagate(sat: CPM_pysat, I=[], user_vars=None, verbose=True):
    """
        optimal_propagate produces the intersection of all models of cnf more precise
        projected on focus.

        Args:
        sat (list): sat solver instantiated with assumption literals to activate or
                de-active constraints.

        I (set): partial model of sat literals

        user_vars (list):
            +/- selected boolean variables of the sat solver
        """
    solved = sat.solve(assumptions=I)

    if not solved:
        raise Exception("UNSAT assumptions")

    # mapping to the user variables
    sat_model = set(lit if lit._value else ~lit for lit in sat.user_vars if lit in user_vars)

    if verbose:
        print("1st sat model:", [lit for lit in sat_model])

    bi = boolvar()

    while(True):
        # negate the values of the model
        sat += bi.implies( Operator("or",[~lit for lit in sat_model]))
        if verbose:
            print("\n\t implies:", [~lit for lit in sat_model])
        solved = sat.solve(assumptions=I | set([bi]))

        if not solved:
            sat += (~bi)
            return set(lit for lit in sat_model)

        # project onto sat model
        new_sat_model = set(lit if lit._value else ~lit for lit in sat.user_vars)
        sat_model = sat_model & new_sat_model
        if verbose:
            print("\n\t new sat model:", new_sat_model)
            print("\n\t Intersection sat model:", sat_model)


def explain_one_step_ocus(sat, soft_weights, Iend, I, verbose=False):
    """
    # Optimisation model:

    The constrained optimal hitting set is described by:

    - x_l={0,1} is a boolean decision variable if the literal is selected
                or not.
    - w_l=f(l) is the cost assigned to having the literal in the hitting
                set (INF otherwise).
    - c_lj={0,1} is 1 (0) if the literal l is (not) present in hitting set j.

    Objective:
            min sum(x_l * w_l) over all l in I + (-Iend \ -I)
    Subject to:
        (1) sum x_l * c_lj >= 1 for all hitting sets j.
            = Hitting set must hit all sets-to-hit.

        (2) sum x_l == 1 for l in (-Iend \ -I)
            = exactly 1 literal explained at a time
    Args:
        sat (pysat.solver): SAT solver instantiated with hard constraints
                            to check for satisfiability of subset.

        Iend (set): Cautious consequence, the set of literals that hold in
                    all models.

        I (set): partial interpretation subset of Iend.
    """
    hs_vars = boolvar(shape=len(Iend))
    F = I | set(~var for var in Iend if var not in I)
    remaining_hs_vars = hs_vars[[id for id, var in enumerate(F) if var not in I]]

    # mapping and back
    hs_vars_Iend = {}

    for id, var in enumerate(F):
        hs_vars_Iend[hs_vars[id]] = var
        hs_vars_Iend[var] = hs_vars[id]

    # conditional optimisation model
    hs_model = Model(
        # exactly one variable to explain!
        sum(remaining_hs_vars) == 1,
        # optimal hitting set
        minimize=sum(hs_var * soft_weights[hs_vars_Iend[hs_var]] for hs_var in  hs_vars)
    )

    ## instantiate hitting set solver
    hittingset_solver = CPM_ortools(hs_model)

    while(True):
        hittingset_solver.solve()
        hs = [hs_var for hs_var in hs_vars if hs_var.value()]

        if not sat.solve(assumptions=[hs_vars_Iend[hs_var] for hs_var in hs]):
            if verbose:
                print("OCUS found !!!!")
                print("hs=", [hs_vars_Iend[hs_var] for hs_var in hs])

            # deleting the hitting set solver
            del hittingset_solver
            return set(hs_vars_Iend[hs_var] for hs_var in hs)

        # hit new hs
        # sum x[j] * hij >= 1
        C = [hs_vars_Iend[Iend_var] for Iend_var in F if not Iend_var.value()]
        hittingset_solver += (sum(C) >= 1)

        if verbose:
            print("hs", hs, [hs_vars_Iend[hs_var] for hs_var in hs])
            print("C=", C)

def explain_ocus(user_vars: set(), soft_weights=None,  hard_constraints=[], given=set(), verbose=False):
    '''
        A SAT-based explanation sequence generating algorithm using assumption
        variables and weighted unsat core extraction.
    '''
    # example cost function
    sat = CPM_pysat(Model(hard_constraints))

    # adding the assumption variables
    I = set(given)

    Iend = optimal_propagate(sat, I=I, user_vars=user_vars, verbose=verbose)

    explanation_sequence = []

    while(I != Iend):
        # explain 1 step using ocus
        ocus_expl = explain_one_step_ocus(sat, soft_weights, Iend, I, verbose)

        # project on known facts and constraints
        used = I & ocus_expl

        # propagate and find information hold in all models
        derived = optimal_propagate(sat, I=used, user_vars=user_vars, verbose=verbose) - I

        # Add newly derived information
        I |= derived

        if verbose:
            print("\t Constraints and literals =", used)
            print("\t Derived", derived)
            print("\t cost=",sum(soft_weights[bv] for bv in ocus_expl) )

        explanation = {
            "constraints": list(used),
            "derived": list(derived),
            "cost": sum(soft_weights[bv] for bv in ocus_expl)
        }

        explanation_sequence.append(explanation)

    # return the list of original (non-flattened) constraints
    return explanation_sequence

def frietkot_explain():
    # Construct the model.
    mayo = boolvar(name="mayo")
    ketchup = boolvar(name="ketchup")
    curry = boolvar(name="curry")
    andalouse = boolvar(name="andalouse")
    samurai = boolvar(name="samurai")

    all_vars = (mayo, ketchup, curry, andalouse, samurai)

    # Pure CNF
    Nora = mayo | ketchup
    Leander = ~samurai | mayo
    Benjamin = ~andalouse | ~curry | ~samurai
    Behrouz = ketchup | curry | andalouse
    Guy = ~ketchup | curry | andalouse
    Daan = ~ketchup | ~curry | andalouse
    Celine = ~samurai
    Anton = mayo | ~curry | ~andalouse
    Danny = ~mayo | ketchup | andalouse | samurai
    Luc = ~mayo | samurai

    allwishes   = [Nora, Leander, Benjamin, Behrouz, Guy, Daan, Celine, Anton, Danny, Luc]

    assum_Nora = boolvar(name="Nora")
    assum_Leander = boolvar(name="Leander")
    assum_Benjamin = boolvar(name="Benjamin")
    assum_Behrouz = boolvar(name="Behrouz")
    assum_Guy = boolvar(name="Guy")
    assum_Daan = boolvar(name="Daan")
    assum_Celine = boolvar(name="Celine")
    assum_Anton = boolvar(name="Anton")
    assum_Danny = boolvar(name="Danny")
    assum_Luc = boolvar(name="Luc")

    ind = [assum_Nora, assum_Leander, assum_Benjamin, assum_Behrouz, assum_Guy, assum_Daan, assum_Celine, assum_Anton, assum_Danny, assum_Luc]

    wish_weights = [40, 40, 60, 60, 60, 60, 20, 60, 80, 40]

    # make assumption indicators, add reified constraints
    assum_allwishes = []
    indcosts = {}

    for i,bv in enumerate(ind):
        assum_allwishes += [bv.implies(allwishes[i])]
        indcosts[bv] = wish_weights[i]
        indcosts[~bv] = wish_weights[i]

    for bv in all_vars:
        indcosts[bv] = 1
        indcosts[~bv] = 1

    # to map indicator variable back to soft_constraints
    indmap = dict((v,i) for (i,v) in enumerate(ind))

    user_vars = set(var for var in all_vars) | set(var for var in ind)
    user_vars |= set(~var for var in user_vars)

    explanation_sequence = explain_ocus(user_vars, indcosts, hard_constraints=assum_allwishes, given=set(var for var in ind))
    print_explanations(explanation_sequence, soft_constraints=allwishes, indmap=indmap)

if __name__ == '__main__':
    frietkot_explain()
