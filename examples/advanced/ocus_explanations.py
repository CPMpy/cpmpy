from cpmpy.solvers.ortools import CPM_ortools
from cpmpy.solvers.pysat import CPM_pysat
from cpmpy.expressions.core import Operator
from cpmpy import *

import numpy as np

def frietkot_explain(verbose=False):
    # Construct the model.
    mayo = boolvar(name="mayo")
    ketchup = boolvar(name="ketchup")
    curry = boolvar(name="curry")
    andalouse = boolvar(name="andalouse")
    samurai = boolvar(name="samurai")

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

    wish_weights = [40, 40, 60, 60, 60, 60, 20, 60, 80, 40]

    explanation_sequence = explain_ocus(allwishes, wish_weights, hard=[], verbose=verbose)
    print(explanation_sequence)

def explain_ocus(soft, soft_weights=None,  hard=[], solver="ortools", verbose=False):
    '''
        A SAT-based explanation sequence generating algorithm using assumption
        variables and weighted unsat core extraction.
    '''
    reified_soft = []
    ind = BoolVar(shape=len(soft), name="ind")

    ## adding assumption variables to the soft constraints
    for i,bv in enumerate(ind):
        reified_soft += [bv.implies(soft[i])]

    # to map indicator variable back to soft_constraints
    indmap = dict((v,i) for (i,v) in enumerate(ind))

    # adding the assumption variables
    I = set(bv for bv in ind)

    Iend = optimal_propagate(hard + reified_soft, I, solver, verbose)

    cost = cost_func(indmap, soft_weights)

    if verbose:
        print("\nMAXIMAL CONSEQUENCE\n\t", Iend)
        print("\nREMAINING TO EXPLAIN\n\t", Iend-I)

    explanation_sequence = []

    while(I != Iend):
        # explain 1 step using ocus
        ocus_expl = explain_one_step_ocus(hard+reified_soft, cost, Iend, I, solver, verbose)

        # project on known facts and constraints
        soft_used = I & ocus_expl

        # propagate and find information hold in all models
        derived = optimal_propagate(hard + reified_soft, soft_used, solver, verbose) - I

        # Add newly derived information
        I |= derived

        explanation = {
            "constraints": list(soft[indmap[con]] if con in indmap else con for con in soft_used),
            "derived": list(derived),
            "cost": sum(cost(con) for con in ocus_expl)
        }

        explanation_sequence.append(explanation)

    # return the list of original (non-flattened) constraints
    return explanation_sequence

def explain_one_step_ocus(hard, cost, Iend, I, solver="ortools", verbose=False):
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

    sat = SolverLookup.lookup(solver)(Model(hard))

    ## OPT Model: Variables
    hs_vars = boolvar(shape=len(Iend))
    F = I | set(~var for var in Iend - I)

    # id of variables that need to be explained
    remaining_hs_vars = hs_vars[[id for id, var in enumerate(F) if var not in I]]

    # mapping between hitting set variables hs_var <-> Iend
    hs_vars_to_Iend = dict( (hs_vars[id], var) for id, var in enumerate(F))
    Iend_to_hs_vars = dict( (var, hs_vars[id]) for id, var in enumerate(F))


    # CONDITIONAL OPTIMISATION MODEL
    hs_model = Model(
        # exactly one variable to explain!
        sum(remaining_hs_vars) == 1,
        # optimal hitting set
        minimize=sum(hs_var * cost(hs_vars_to_Iend[hs_var]) for hs_var in hs_vars)
    )

    ## instantiate hitting set solver
    hittingset_solver = SolverLookup.lookup(solver)(hs_model)

    while(True):
        hittingset_solver.solve()

        # Get hitting set
        hs = hs_vars[hs_vars.value() == 1]

        # map to vars of formula F
        S = set(hs_vars_to_Iend[hs_var] for hs_var in hs)

        if verbose:
            print("\t hs", hs, S)

        # SAT check and computation of model
        if not sat.solve(assumptions=S):
            if verbose:
                print("OCUS=", S)

            # deleting the hitting set solver
            return S

        # compute complement of model in formula F
        C =  [Iend_to_hs_vars[Iend_var] for Iend_var in F if not Iend_var.value()]

        # Add complement as a new set to hit: sum x[j] * hij >= 1
        hittingset_solver += (sum(C) >= 1)

        if verbose:
            print("C=", C)

def optimal_propagate(hard, soft, solver="ortools", verbose=False):
    """
        optimal_propagate produces the intersection of all models of cnf more precise
        projected on focus.

        Args:
        hard (list): List of hard constraints

        soft (iterable): List of soft constraints

        I (set): partial model of sat literals

        user_vars (list):
            +/- selected boolean variables of the sat solver
    """
    sat = SolverLookup.lookup(solver)(Model(hard))
    assert sat.solve(assumptions=soft), "Propagation of soft constraints only possible if model is SAT."

    # initial model that needs to be refined
    sat_model = set(lit if lit.value() else ~lit for lit in sat.user_vars)

    if verbose:
        print("Initial sat model:", sat_model)

    while(True):
        # negate the values of the model
        blocking_clause = any([~lit for lit in sat_model])
        if verbose:
            print("\n\tBlocking clause:", blocking_clause)

        sat += blocking_clause

        solved = sat.solve(assumptions=soft)

        if not solved:
            return set(lit for lit in sat_model)

        new_sat_model = set(lit if lit.value() else ~lit for lit in sat.user_vars)

        # project new model onto sat model
        sat_model = sat_model & new_sat_model
        if verbose:
            print("\n\t new sat model:", new_sat_model)
            print("\n\t Intersection sat model:", sat_model)

def cost_func(indmap, soft_weights):
    '''
        Example cost function with mapping of indicator constraints to
        corresponding given weight.

        Variables not in the indicator map have a unit weight, which
        corresponds to using a unit boolean variable.
    '''
    def cost_lit(var):
        if var in indmap:
            return soft_weights[indmap[var]]
        else:
            return 1

    return cost_lit

if __name__ == '__main__':
    frietkot_explain(verbose=False)
