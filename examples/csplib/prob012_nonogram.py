"""
    Nonogram in CPMpy

    Problem 012 on CSPlib
    https://www.csplib.org/Problems/prob012/

    Nonograms are a popular puzzle, which goes by different names in different countries.
    Solvers have to shade in squares in a grid so that blocks of consecutive shaded squares satisfy constraints given for each row and column.
    Constraints typically indicate the sequence of shaded blocks (e.g. 3,1,2 means that there is a block of 3, then a gap of unspecified size, a block of length 1, another gap, and then a block of length 2).

    Using native solver access to OR-tools to post Automaton constraint

    Based on model by Hakank

    Model created by Ignace Bleukx, ignace.bleukx@kuleuven.be

"""
import sys
import json
import requests

import numpy as np

from cpmpy import *

def nonogram(row_rules, col_rules):

    solver = SolverLookup.get("ortools")

    n_rows, n_cols = len(row_rules), len(col_rules)
    board = intvar(0,1,shape=(n_rows,n_cols), name="board")
    solver.user_vars.update(set(board.flatten()))

    # patterns of each row must be correct
    for r, pattern in enumerate(row_rules):
        automaton_func, final_states = transition_function(pattern)
        solver.ort_model.AddAutomaton(
            solver.solver_vars(board[r]),
            starting_state=0, final_states=final_states,
            transition_triples = automaton_func
        )

    # patterns of each column must be correct
    for c, pattern in enumerate(col_rules):
        automaton_func, final_states = transition_function(pattern)
        solver.ort_model.AddAutomaton(
            solver.solver_vars(board[:,c]),
            starting_state=0, final_states=final_states,
            transition_triples = automaton_func
        )

    return solver, (board,)


def transition_function(pattern):
    """
        Pattern is a vector containing the lengths of blocks with value 1
    """
    func = []
    n_states = 0
    for block_length in pattern:
        if block_length == 0:
            continue
        func += [(n_states, 0, n_states)]
        for _ in range(block_length):
            func += [(n_states, 1, n_states+1)]
            n_states += 1

        func += [(n_states, 0, n_states+1)]
        n_states += 1

    func += [(n_states, 0, n_states)]
    # line can end with 0 or 1
    return func, [n_states-1,n_states]

def get_data(data, name):
    for entry in data:
        if entry.name == name:
            return entry


if __name__ == "__main__":

    fname = "https://raw.githubusercontent.com/CPMpy/cpmpy/csplib/examples/csplib/prob012_nonogram.json"
    problem_name = "lambda"

    data = None

    if len(sys.argv) > 1:
        fname = sys.argv[1]
        with open(fname, "r") as f:
            data = json.load(f)

    if len(sys.argv) > 2:
        problem_name = sys.argv[2]

    if data is None:
        data = requests.get(fname).json()

    rules = get_data(data, problem_name)

    model, (board,) = nonogram(**rules)

    if model.solve():
        f = {"int":lambda x : " " if x == 0 else "#"}
        print(np.array2string(board.value(), formatter=f))
    else:
        print("Model is unsatisfiable!")