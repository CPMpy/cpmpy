from cpmpy import *
from cpmpy.transformations.flatten_model import flatten_constraint
from cpmpy.transformations.get_variables import get_variables_model
import numpy as np

def sol_intersect(model):
    # prep
    model_vars = cpm_array(get_variables_model(model))

    in_intersect = np.ones(model_vars.shape, dtype=bool)
    # first sol
    assert model.solve(), "Model should be satisfiable!"

    vals = model_vars.value()

    # next, find a non-overlapping one
    model += any(model_vars[in_intersect] != vals[in_intersect])

    # in_intersect will shrink per iteration
    while model.solve():
        in_intersect = (model_vars.value() == vals)
        # next, find a non-overlapping one
        model += any(model_vars[in_intersect] != vals[in_intersect])
    return model_vars[in_intersect] == vals[in_intersect]

def sudoku_model(given, verbose=False):
    e = 0
    DIFFICULTY_ROW_CONSTRAINT = 60
    DIFFICULTY_COL_CONSTRAINT = 60
    DIFFICULTY_BLOCK_CONSTRAINT = 60
    DIFFICULTY_ASSIGNMENT = 1


    n = given.shape[0]

    puzzle = intvar(1,n, shape=(n, n), name="puzzle")

    model = Model()

    # Constraints on rows and columns
    soft_weights = []
    assum_lits = []
    
    for row in puzzle:
        lit = boolvar()
        model += (lit == AllDifferent(row))
        assum_lits.append(lit)

    soft_weights += n *[DIFFICULTY_ROW_CONSTRAINT]

    for col in puzzle.T:
        lit = boolvar()
        model += (lit == AllDifferent(col))
        assum_lits.append(lit)

    soft_weights += n *[DIFFICULTY_COL_CONSTRAINT]

    # Constraints on values (cells that are not empty)
    for con in puzzle[given!=e] == given[given!=e]:
        lit = boolvar()
        model += (lit == con) # numpy's indexing, vectorized equality
        assum_lits.append(lit)

    soft_weights += len(given[given!=e]) * [DIFFICULTY_ASSIGNMENT]

    # Constraints on blocks
    root = int(n **(1/2))
    for i in range(0,n, root):
        for j in range(0,n, root):
            lit = boolvar()
            model += (lit == AllDifferent(puzzle[i:i+root, j:j+root])) # python's indexing
            assum_lits.append(lit)

    soft_weights += n *[DIFFICULTY_BLOCK_CONSTRAINT]
    return model, assum_lits, soft_weights, puzzle

def main(verbose=False):
    e = 0

    given = np.array([
        [e, 1, 2, 3],
        [2, e, e, e],
        [3, e, e, 1],
        [1, e, 3, e]
    ])

    model, assum_lits, soft_weights, puzzle  = sudoku_model(given, verbose)
    print(model)
    explanation_sequence = explain_sol(model + assum_lits, soft_weights=soft_weights, verbose=verbose)
    print("Explnaation sequence=", explanation_sequence)


def explain_sol(model, soft_weights=None, verbose=True):
    explanation_sequence = []

    if soft_weights is None:
        soft_weights = [1] * len(model.constraints)

    solution = sol_intersect(model)

    return explanation_sequence

if __name__ == '__main__':
    main(verbose=True)
    