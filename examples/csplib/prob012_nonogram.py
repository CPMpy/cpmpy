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

def nonogram(row_rules, col_rules,**kwargs):

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


# Helper functions
def _get_instance(data, pname):
    for entry in data:
        if pname == entry["name"]:
            return entry
    raise ValueError(f"Problem instance with name '{pname}' not found, use --list-instances to get the full list.")


def _print_instances(data):
    import pandas as pd
    df = pd.json_normalize(data)
    df_str = df.to_string(columns=["name", "rows", "cols"], na_rep='-')
    print(df_str)


if __name__ == "__main__":
    import argparse
    import json
    import requests

    # argument parsing
    url = "https://raw.githubusercontent.com/CPMpy/cpmpy/csplib/examples/csplib/prob012_nonogram.json"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-instance', nargs='?', default="turing", help="Name of the problem instance found in file 'filename'")
    parser.add_argument('-filename', nargs='?', default=url, help="File containing problem instances, can be local file or url")
    parser.add_argument('--list-instances', help='List all problem instances', action='store_true')

    args = parser.parse_args()

    if "http" in args.filename:
        problem_data = requests.get(args.filename).json()
    else:
        with open(args.filename, "r") as f:
            problem_data = json.load(f)

    if args.list_instances:
        _print_instances(problem_data)
        exit(0)

    problem_params = _get_instance(problem_data, args.instance)
    print("Problem name:", problem_params["name"])

    model, (board,) = nonogram(**problem_params)

    if model.solve():
        np.set_printoptions(threshold=np.inf, linewidth=1024)
        f = {"int":lambda x : " " if x == 0 else chr(0x2588)}
        print(np.array2string(board.value(), formatter=f))
    else:
        print("Model is unsatisfiable!")