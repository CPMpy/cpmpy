import cpmpy
from cpmpy import *
from cpmpy.expressions.core import Comparison
from cpmpy.transformations.flatten_model import flatten_constraint, flatten_model
from cpmpy.transformations.get_variables import get_variables_model
import numpy as np

EMPTY = 0
DIFFICULTY_ROW_CONSTRAINT = 60
DIFFICULTY_COL_CONSTRAINT = 60
DIFFICULTY_BLOCK_CONSTRAINT = 60
DIFFICULTY_ASSIGNMENT = 1

def equals(c1, c2):
    if set(c1.args) == set(c2.args) and c1.name == c2.name:
        return True
    else:
        return False

def sol_intersect(model, model_vars):

    in_intersect = np.ones(model_vars.shape, dtype=bool)
    # first sol
    assert model.solve(), "Model should be satisfiable!"
    vals = model_vars.value()
    # next, find a non-overlapping one
    print(model_vars[in_intersect])
    print(vals[in_intersect])

    model += any(model_vars[in_intersect] != vals[in_intersect])

    # in_intersect will shrink per iteration
    while model.solve():
        in_intersect = (model_vars.value() == vals)
        # next, find a non-overlapping one
        model += any(model_vars[in_intersect] != vals[in_intersect])
    
    return model_vars[in_intersect] == vals[in_intersect]

def sudoku_model(given, verbose=False):
    n = given.shape[0]

    puzzle = intvar(1,n, shape=(n, n), name="puzzle")

    cons = []

    # Constraints on rows and columns
    soft_weights = []

    for row in puzzle:
        cons.append(AllDifferent(row))

    soft_weights += n * [DIFFICULTY_ROW_CONSTRAINT]

    for col in puzzle.T:
        cons.append(AllDifferent(col))

    soft_weights += n * [DIFFICULTY_COL_CONSTRAINT]

    # Constraints on blocks
    root = int(n **(1/2))
    for i in range(0,n, root):
        for j in range(0,n, root):
            cons.append(AllDifferent(puzzle[i:i+root, j:j+root])) # python's indexing

    soft_weights += n *[DIFFICULTY_BLOCK_CONSTRAINT]

    assert len(cons) == len(soft_weights), f"# Constraints [{len(cons)}] == # soft_weights [{len(soft_weights)}]"
    return cons, soft_weights, puzzle

def ocus(soft_constraints, soft_weights, hard_constraints=[], solver='ortools', verbose=False):
    # small optimisation: pre-flatten all constraints once
    # so it needs not be done over-and-over in solving
    hard = flatten_constraint(hard_constraints) # batch flatten
    soft = [flatten_constraint(c) for c in soft_constraints]

    ## Mip model
    if Model(hard+soft).solve():
        if verbose:
            print("Unexpectedly, the model is SAT")
        return []

    hs_vars = boolvar(shape=len(soft_constraints), name="hs_vars")

    hs_mip_model = Model(
        # Objective: min sum(x_l * w_l)
        minimize=sum(var * soft_weights[id] for id, var in enumerate(hs_vars))
    )

    # instantiate hitting set solver
    hittingset_solver = SolverLookup.lookup(solver)(hs_mip_model)

    while(True):
        hittingset_solver.solve()

        # Get hitting set
        hs_soft = [soft[id] for id, hs_var in enumerate(hs_vars) if hs_var.value() == 1]
        if verbose>2:
            print('\n\ths=', hs_soft)

        if not Model(hard+hs_soft).solve():
            if verbose > 1:
                print("\n\t ===> OCUS =", hs_soft)

            return [soft_constraints[id] for id, hs_var in enumerate(hs_vars) if hs_var.value() == 1]

        # compute complement of model in formula F
        C = hs_vars[hs_vars.value() != 1]

        # Add complement as a new set to hit: sum x[j] * hij >= 1
        hittingset_solver += (sum(C) >= 1)

        if verbose > 2:
            print("\t Complement =", C)

def explain_one_step(model, soft_weights, partial_sol, sol):
    expl = set()
    remaining_to_explain = []
    for iend in sol:
        contained = False
        for i in I:
            if equals(i, iend):
                contained = True
        if not contained:
            remaining_to_explain.append(iend)

    print("\n", "Explain One Step","\n")
    print(model.constraints)

    for lit in remaining_to_explain[:1]:
        print("\n", "\t Explaining ", lit)
        negated_literal = lit.args[0] != lit.args[1]
        assert((len(model.constraints) + 1) == len(soft_weights)), f"({len(model.constraints) + 1}) == {len(soft_weights)}"
        new_model = Model(*model.constraints) + negated_literal
        print(*new_model.constraints)
        print(ocus(new_model.constraints, soft_weights=soft_weights, hard_constraints=[], solver='ortools', verbose=2))   

    #print(ocus(model.constraints, soft_weights=f, hard_constraints=[], solver='ortools', verbose=False))
    return expl

def explain(model, model_vars, given, soft_weights=None, verbose=True):
    # Defaulting to unit costs

    if soft_weights is None:
        soft_weights = [1] * len(model.constraints)

    # Maximal consequence of given constraints
    if len(soft_weights) == len(model.constraints):
        soft_weights += [1] * len(given)

    current_assignment = [con for con in given]

    new_model = flatten_model(model + current_assignment)

    solution = list(sol_intersect(Model(*new_model.constraints), model_vars))

    explanation_sequence = []
    nsteps = 1
    step_count= 0

    while(len(current_assignment) != len(solution) and step_count < nsteps):

        ocus_expl = explain_one_step(new_model, soft_weights + [1], I=current_assignment, Iend=solution)

        used = [i for i in ocus_expl if i in new_model.constraints]
        facts = [i for i in ocus_expl if i in current_assignment]
        derived = set()
        #derived = sol_intersect(Model(used + facts), model_vars)
        cost = sum(soft_weights[id] for id, c in enumerate(model.constraints) if c in used) + len(facts)

        explanation_sequence.append({
            'constraints': used,
            'facts': used,
            'derived': derived,
            'cost': cost
        })
        step_count += 1

    print(solution)

    return explanation_sequence

def main(verbose=False):

    given = np.array([
        [EMPTY, 1,      2,      3],
        [2,     EMPTY,  EMPTY,  EMPTY],
        [3,     EMPTY,  EMPTY,  1],
        [1,     EMPTY,  3,      EMPTY]
    ])

    soft_constraints, soft_weights, puzzle  = sudoku_model(given, verbose)
    print(soft_constraints)

    # Constraints on values (cells that are not empty)
    I = [con for con in puzzle[given!=EMPTY] == given[given!=EMPTY]]
    print(I)
    print(I + soft_constraints)

    explanation_sequence = explain(soft_constraints, model_vars=puzzle, given=I, soft_weights=soft_weights, verbose=verbose)

    print("Explnaation sequence=", explanation_sequence)


if __name__ == '__main__':
    main(verbose=True)
    