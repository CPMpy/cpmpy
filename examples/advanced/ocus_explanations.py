from cpmpy.solvers.ortools import CPM_ortools
from cpmpy.solvers.pysat import CPM_pysat
from cpmpy.expressions.core import Operator
from cpmpy import *
from cpmpy.transformations.get_variables import get_variables_model

import numpy as np

def frietkot_explain(verbose=False):
    '''
        The 'frietkot' problem; invented by Tias to explain SAT and SAT solving
        http://homepages.vub.ac.be/~tiasguns/frietkot/
    '''
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

    if verbose > 0:
        print("\nExplanation Sequence:")
        print("-" * len("Explanation Sequence:"))
        for id, explanation in enumerate(explanation_sequence):
            print(f"\n{id}. Using Constraints:\t", explanation["constraints"])
            print("\t     Derived:\t", explanation["derived"])
            print("\t        Cost:\t", explanation["cost"])

    return explanation_sequence

def explain_ocus(soft, soft_weights=None,  hard=[], solver="ortools", verbose=False):
    '''
        A SAT-based explanation sequence generating algorithm using assumption
        variables and weighted UNSAT core (Optimal Constrained Unsatisfiable Subset)
        extraction. [1, 2]

        [1] Bogaerts, B., Gamba, E., Claes, J., & Guns, T. (2020). Step-wise explanations
        of constraint satisfaction problems. In ECAI 2020-24th European Conference on
        Artificial Intelligence, 29 August-8 September 2020, Santiago de Compostela,
        Spain, August 29-September 8, 2020-Including 10th Conference on Prestigious
        Applications of Artificial Intelligence (PAIS 2020) (Vol. 325, pp. 640-647).
        IOS Press; https://doi. org/10.3233/FAIA200149.

        [2] Bogaerts, B., Gamba, E., & Guns, T. (2021). A framework for step-wise explaining
        how to solve constraint satisfaction problems. Artificial Intelligence, 300, 103550.
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

    # Cost function returns cost of soft weight if a constraint is used
    # otherwhise returns 1 (i.e. using a literal)
    cost = cost_func(indmap, soft_weights)

    if verbose > 0:
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
        Optimal Constrained Unsatisfiable Subsets (OCUS) for CSP explanations [1]

        explain_one_step_ocus relies on the implicit hitting set duality between
        MUSes and MCSes for a given formula F [2, 3]:

            - A set S \subseteq of F is an MCS of F iff it is a minimal hitting set
             of MUSs(F).
            - A set S \subseteq F is a MUS of F iff it is a minimal hitting set
             of MCSs(F).

        Builds a MIP model for computing minimal (optimal) hitting sets. Repeatedly
        checks for satisfiability until UNSAT, which means an OCUS is found.
        If SAT, and sat solver computes a model. The complement w.r.t F is
        computed and added as a new set-to-hit.

        MIP MODEL
        ---------

        The constrained optimal hitting set is described by:

            - x_l={0,1} is a boolean decision variable if the literal is inside the
                        hitting set or not.
            - w_l=f(l) is the cost assigned to having the literal in the hitting
                        set.
            - c_lj={0,1} is 1 (0) if the literal l is (not) present in the set-to-hit j.

        Objective:
                min sum(x_l * w_l) over all l in I + (-Iend \ -I)

        Subject to:
            (1) sum x_l * c_lj >= 1 for all hitting sets j.
                = The hitting set must hit all sets-to-hit.

            (2) sum x_l == 1 for l in (-Iend \ -I)
                = exactly 1 literal explained at a time

        Args
        ----

            hard (list[cpmpy.Expression]): Hard constraints

            Iend (set): Cautious consequence, the set of literals that hold in
                        all models.

            I (set): partial interpretation (subset of Iend).

        [1] Gamba, E., Bogaerts, B., & Guns, T. (8 2021). Efficiently Explaining CSPs
        with Unsatisfiable Subset Optimization. In Z.-H. Zhou (Red), Proceedings of the
        Thirtieth International Joint Conference on Artificial Intelligence,
        IJCAI-21 (bll 1381–1388). doi:10.24963/ijcai.2021/191.

        [2] Liffiton, M. H., & Sakallah, K. A. (2008). Algorithms for computing minimal
        unsatisfiable subsets of constraints. Journal of Automated Reasoning, 40(1), 1-33.

        [3] Reiter, R. (1987). A theory of diagnosis from first principles.
        Artificial intelligence, 32(1), 57-95.
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

        if verbose > 0:
            print("\n\t hs =", hs, S)

        # SAT check and computation of model
        if not sat.solve(assumptions=S):
            if verbose > 0:
                print("\n\t ===> OCUS =", S)

            # deleting the hitting set solver
            return S
        # satisfying model
        S = set(v == v.value() for v in F)

        # compute complement of model in formula F
        C =  F - S
        set_to_hit = set(Iend_to_hs_vars[Iend_var] for Iend_var in F - S)

        # Add complement as a new set to hit: sum x[j] * hij >= 1
        hittingset_solver += (sum(set_to_hit) >= 1)

        if verbose > 0:
            print("\t Complement =", C)
            print("\t set-to-hit =", set_to_hit)

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
    # Build sat model
    sat_model = Model(hard)
    sat = SolverLookup.lookup(solver)(sat_model)
    assert sat.solve(assumptions=soft), "Propagation of soft constraints only possible if model is SAT."

    # Extracting only relevant variables
    user_vars = get_variables_model(sat_model)

    # initial model that needs to be refined
    sat_model = set(v == v.value() for v in user_vars)

    if verbose > 1:
        print("\nOptimal Propagate")
        print("-" * len("Optimal Propagate"))
        print("\nInitial sat model:", sat_model)

    while(True):
        # negate the values of the model
        blocking_clause = ~all(v == v.value() for v in sat_model)
        if verbose > 1:
            print("\n\tBlocking clause:", blocking_clause)

        sat += blocking_clause

        solved = sat.solve(assumptions=soft)

        if not solved:
            return sat_model

        new_sat_model = set(v == v.value() for v in user_vars)

        # project new model onto sat model
        sat_model = sat_model & new_sat_model

        if verbose > 1:
            print("\n\t new sat model:", new_sat_model)
            print("\n\t Intersection with previous sat model:", sat_model)

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
    frietkot_explain(verbose=1)
