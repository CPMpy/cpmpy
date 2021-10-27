from cpmpy import *
from cpmpy.transformations.get_variables import get_variables_model

import numpy as np

def frietkot_explain(verbose=0):
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

    return explanation_sequence

def explain_ocus(soft, soft_weights=None,  hard=[], solver="ortools", verbose=0):
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
    curr_sol = set()
    # compute all derivable literals
    full_sol = solution_intersection(Model(hard + soft), solver, verbose=verbose)


    # prep soft constraint formulation with a literal for each soft constraint
    # (so we can iteratively use an assumption solver on softliterals)
    soft_lit = BoolVar(shape=len(soft), name="ind")
    reified_soft = []
    for i,bv in enumerate(soft_lit):
        reified_soft += [bv.implies(soft[i])]
    # to map indicator variable back to soft_constraints
    indmap = dict((v,i) for (i,v) in enumerate(soft_lit))

    if soft_weights is None:
        soft_weights = 5*np.ones(len(soft), dtype=int)
    # Cost function returns cost of soft weight if a constraint is used
    # otherwhise returns 1 (i.e. using a literal)
    cost = cost_func(list(soft_lit), soft_weights)

    if verbose > 0:
        if len(curr_sol):
            print("Current solution:", curr_sol)
        print("Solution intersection:", full_sol-curr_sol)
        print("") # blank line

    
    # we will explain literal by literal
    explanation_sequence = []
    while(curr_sol != full_sol):
        # explain 1 step using ocus
        remaining_to_explain = full_sol - curr_sol
        all_lit = set(soft_lit) | curr_sol

        ocus_expl = explain_one_step_ocus(hard + reified_soft, all_lit, cost, remaining_to_explain, solver, verbose)

        # project on known facts and constraints
        cons_used =  [soft[indmap[con]] for con in set(soft_lit) & ocus_expl]
        facts_used = curr_sol & ocus_expl
        derived = set(~v for v in ocus_expl - set(soft_lit) - facts_used)

        # Add newly derived information
        curr_sol |= derived

        explanation = {
            "constraints": list(cons_used),
            "facts": list(facts_used),
            "derived": list(derived),
            "cost": sum(cost(con) for con in ocus_expl)
        }

        explanation_sequence.append(explanation)

        if verbose > 0:
            print(f"Constraint(s): {explanation['constraints']}")
            print(f"  and fact(s): {explanation['facts']}")
            print(f"           ==> {explanation['derived'][0]}\t(cost: {explanation['cost']})")
            print("") # blank line

    # return the list of original (non-flattened) constraints
    return explanation_sequence

def explain_one_step_ocus(hard, soft_lit, cost, remaining_sol_to_explain, solver="ortools", verbose=False):
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
    ## Unsatisfiable Formula = soft constraints + (~remaining_to_explain)
    neg_remaining_sol_to_explain = set(~var for var in remaining_sol_to_explain)
    F = set(soft_lit) | neg_remaining_sol_to_explain


    ## ----- CONDITIONAL OPTIMISATION MODEL------
    ## -------------- VARIABLES -----------------
    hs_vars = boolvar(shape=len(soft_lit) + len(remaining_sol_to_explain))

    # id of variables that need to be explained
    remaining_hs_vars = hs_vars[[id for id, var in enumerate(F) if var in neg_remaining_sol_to_explain]]
    # mapping between hitting set variables hs_var <-> Iend
    varmap_hs_sat = dict( (hs_vars[id], var) for id, var in enumerate(F))
    varmap_sat_hs = dict( (var, hs_vars[id]) for id, var in enumerate(F))

    ## ----------------- MODEL ------------------
    hs_mip_model = Model(
         # exactly one variable to explain!
        sum(remaining_hs_vars) == 1,
        # Objective: min sum(x_l * w_l) over all l in I + (-Iend \ -I)
        minimize=sum(hs_var * cost(varmap_hs_sat[hs_var]) for hs_var in hs_vars) 
    )

    # instantiate hitting set solver
    hittingset_solver = SolverLookup.lookup(solver)(hs_mip_model)


    ## ----- SAT solver model ----
    SAT = SolverLookup.lookup(solver)(Model(hard))

    while(True):
        hittingset_solver.solve()

        # Get hitting set
        hs = hs_vars[hs_vars.value() == 1]

        # map to vars of formula F
        S = set(varmap_hs_sat[hs_var] for hs_var in hs)

        if verbose > 1:
            print("\n\t hs =", hs, S)

        # SAT check and computation of model
        if not SAT.solve(assumptions=S):
            if verbose > 1:
                print("\n\t ===> OCUS =", S)

            return S

        # satisfying model
        S = set(v if v.value() else ~v for v in F)

        # compute complement of model in formula F
        C =  F - S

        set_to_hit = set(varmap_sat_hs[var] for var in C)

        # Add complement as a new set to hit: sum x[j] * hij >= 1
        hittingset_solver += (sum(set_to_hit) >= 1)

        if verbose > 1:
            print("\t Complement =", C)
            print("\t set-to-hit =", set_to_hit)

def solution_intersection(model, solver="ortools", verbose=0):
    """
        solution_intersection produces the intersection of all models
    """
    # Build sat model
    sat_vars = get_variables_model(model)

    SAT = SolverLookup.lookup(solver)(model)

    assert SAT.solve(), "Propagation of soft constraints only possible if model is SAT."
    sat_model = set(bv if bv.value() else ~bv for bv in sat_vars)

    while(SAT.solve()):
        # negate the values of the model
        sat_model &= set(bv if bv.value() else ~bv for bv in sat_vars)
        blocking_clause = ~all(sat_model)

        SAT += blocking_clause

        if verbose >= 2:
            print("\n\tBlocking clause:", blocking_clause)

    return sat_model


def cost_func(soft, soft_weights):
    '''
        Example cost function with mapping of indicator constraints to
        corresponding given weight.

        Variables not in the indicator map have a unit weight, which
        corresponds to using a unit boolean variable.
    '''

    def cost_lit(cons):
        # return soft weight if constraint is a soft constraint
        if len(set({cons}) & set(soft)) > 0:
            return soft_weights[soft.index(cons)]
        else:
            return 1

    return cost_lit

if __name__ == '__main__':
    frietkot_explain(verbose=1)
