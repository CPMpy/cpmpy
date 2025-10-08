#!/usr/bin/env python
import pytest
import itertools
import numpy as np
import math
import cpmpy as cp
from cpmpy.transformations import int2bool

import random


def log(*mess, env, verbosity=1, end="\n"):
    if verbosity <= env["verbosity"]:
        print(*mess, end=end)


def generate_table_from_example():
    x = cp.intvar(1, 4, name="x")
    y = cp.intvar(1, 3, name="y")
    z = cp.intvar(1, 3, name="z")
    X = (x, y, z)
    T = [(2, 1, 1), (3, 2, 2), (4, 3, 3), (1, 2, 3), (2, 1, 2)]
    T = np.array(T)
    return X, T


def generate_table_from_data(rows, d):
    """Generate a table constraint with the given `rows` and with var domains of size `d`"""
    return cp.intvar(1, d, shape=len(rows[0]), name="x"), np.array(rows)


def generate_table(n, m, d):
    """Generate a table constraint with `n` variables with domains of size `d`, and with `m` rows"""
    X = cp.intvar(1, d, shape=n, name="x")
    T = np.array([tuple(random.randint(1, d) for _ in range(n)) for _ in range(m)])
    return X, T


def encode_x(a, lb, ub):
    X = (ub - lb + 1) * [0]
    X[a - lb] = 1
    return X


def encode(X, T):
    enc = []
    for t in T:
        row_i = []
        for x, row in zip(X, t):
            row_i += encode_x(row, x.lb, x.ub)
        enc += [row_i]
    return np.array(enc)
    return np.array([xij for x, t in zip(X, T) for row in t for xij in encode_x(row, x.lb, x.ub)])


def rows(T, i, j=1):
    """Row indices where T[i]==j"""
    return set(int(i) for i in np.where(T[:, i] == j)[0].flatten())


def cols(T, i, j=1):
    """Col indices where T[i]==j"""
    return set(int(i) for i in np.where(T[i, :] == j)[0].flatten())


def shrink(C, T, env):
    for i in C:
        log(f"shrinking {i} in {C}", env=env, verbosity=3)

        if len(C) <= 1:  # slightly different from
            return C

        S = set.intersection(*[rows(T, l) for l in (C - {i})])
        if len(S) == 0:
            C = C - {i}
    return C


from enum import Enum


class Heuristic(Enum):
    INPUT = 1
    GREEDY = 2
    REDUCE = 3


def choose(A, T_enc, R, env):
    log(f"Choose from {sorted(A)} for table {T_enc} from remaining choices {R}", env=env, verbosity=2)
    if len(A) == 0:
        return None
    match env.get("heuristic", Heuristic.INPUT):
        case Heuristic.INPUT:
            return min(i for i in A)
        case Heuristic.GREEDY:
            return min(A, key=lambda i: len(R.intersection(rows(T_enc, i))))
        case Heuristic.REDUCE:
            choice = min(
                (i for i in A if R.intersection(rows(T_enc, i)) != R),
                key=lambda i: len(rows(T_enc, i)),
                default=None,  # TODO [?] check this edge-case
            )
            return (
                choice if choice is not None else choose(A, T_enc, R, {**env, "heuristic": Heuristic.GREEDY})
            )


# TODO convert T_enc to set of tuples?
def explain(A_enc, T_enc, env):
    log(f"Explain {A_enc}", env=env)

    X = set()  # columns added to cut
    R = set(range(len(T_enc)))  # remaining columns
    W = set(i for i, a in enumerate(A_enc) if a == 1.0)
    F = set(i for i, a in enumerate(A_enc) if 0.0 < a < 1.0)

    log(f"W = {sorted(W)}", verbosity=3, env=env)
    log(f"F = {sorted(F)}", verbosity=3, env=env)

    for iteration in itertools.count(start=1):
        if not len(R):
            break
        if W <= X:
            log("  unexplainable", env=env)
            log(f"    because {W} <= {X}", env=env, verbosity=3)
            return set()  # no cut

        # choose some col which is 1
        log("A", A_enc, X, env=env, verbosity=3)
        i = choose(W - X, T_enc, R, env=env)
        log(f"chosen {i}", env=env, verbosity=3)

        R = R.intersection(rows(T_enc, i))
        X.add(i)

        # Loop termination for debug purposes
        assert iteration <= env.get("max_iterations", iteration)

    log(f"  by explanation of size ({len(X)}): {X}", env=env)
    if env.get("shrink", True):
        size = len(X)
        X = shrink(X, T_enc, env)
        if len(X) < size:
            log(f"  shrunk to size ({len(X)}): {X}", env=env)

    log(f"  cut == {X}", env=env)
    return X


def show_sols(sols, T):
    return ", ".join(f"*{sol}" if list(sol) in T.tolist() else f"{sol}" for sol in sorted(sols))


import gurobipy as gp
from gurobipy import GRB


def assert_integer_solution(A_enc):
    for a_enc_i in A_enc:
        assert math.isclose(a_enc_i, round(a_enc_i), abs_tol=1e-5), (
            f"Expected integer solution for MIP, but got {a_enc_i} in {A_enc}"
        )


def get_x_enc(ivarmap):
    return [x_enc_i for x_enc in ivarmap.values() for x_enc_i in x_enc._xs]


def log_explanation(explanation, frm, env):
    if "cuts" in env:
        if env["debug"]:
            assert explanation not in (c["cut"] for c in env["cuts"]), (
                f"Already found cut {explanation} previously in {env['cuts']}"
            )
        env["cuts"].append({"cut": explanation, "shrunk": 0, "from": frm})


def get_solution_callback(slv, ivarmap, T_enc, env):
    X_enc = get_x_enc(ivarmap)
    X_enc_grb = [slv.solver_var(x_enc_i) for x_enc_i in X_enc]

    def solution_callback(what, where):
        try:
            A_enc = None
            frm = None
            match where:
                case GRB.Callback.MIPNODE:
                    if (  # Optimal solution to LP relaxation
                        what.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL
                    ):
                        A_enc = [what.cbGetNodeRel(x_enc_i) for x_enc_i in X_enc_grb]
                        frm = "MIPNODE-OPT"
                        A_enc = [int(a_enc_i) for a_enc_i in A_enc]
                    else:
                        return
                case GRB.Callback.MIPSOL:  # Integer solution
                    A_enc = [what.cbGetSolution(x_enc_i) for x_enc_i in X_enc_grb]
                    frm = "MIPSOL"
                    assert_integer_solution(A_enc)
                    A_enc = [round(a_enc_i) for a_enc_i in A_enc]
                case _:
                    return

            log(frm, A_enc, env=env, verbosity=2)
            A_enc = [int(a_enc_i) for a_enc_i in A_enc]
            # if constraint.value():
            if A_enc in T_enc.tolist():
                return

            # encode assignment
            explanation = explain(A_enc, T_enc, env)
            log_explanation(explanation, frm, env)

            if not explanation:
                assert not assert_integer_solution(A_enc)
                # TODO assert this was integer solution?
                return  # did not find solution

            grbs = [slv.solver_var(X_enc[c]) for c in explanation]
            what.cbLazy(gp.quicksum(grbs) <= len(grbs) - 1)
            what.write("/tmp/gurobi.lp")
        except Exception as e:
            what._callback_exception = e
            what.terminate()

    return solution_callback


def solve(X, T, env):
    log("X =", ", ".join(f"{x} in {x.lb}..{x.ub}" for x in X), env=env, verbosity=0)
    log("T =", env=env, verbosity=1)
    log(T, env=env, verbosity=1)
    T_enc = encode(X, T)
    log("T_enc =", env=env, verbosity=1)
    log(T_enc, env=env, verbosity=1)

    # Init env
    env["debug"] = env.get("debug", False)
    env["debug_unlucky"] = False  # TODO re-enable
    env["cuts"] = []

    # X_enc = [x == d for x in X for d in range(x.lb, x.ub + 1)]
    ivarmap = {}
    model = cp.Model()
    for x in X:
        x_enc, cons = int2bool._encode_int_var(ivarmap, x, "direct")
        model += [cons]
        expr, k = x_enc.encode_term()
        # model += cp.sum(c * b for c, b in expr) + k == x
    X_enc = get_x_enc(ivarmap)
    model.minimize(sum(X))
    table = cp.Table(X, T)

    sols = []

    if env.get("debug", False):
        n_sols = model.solveAll(
            display=lambda: sols.append([x.value() for x in X]),
            solver=env.get("solver", "gurobi"),
            solution_limit=1000 if env["solver"] == "gurobi" else None,
        )
        assert env["solver"] != "gurobi" or n_sols < 1000

        log(f"Search space remaining: ({len(sols)})", env=env)
        log(show_sols(sols, T), env=env, verbosity=2)
        if len(env["cuts"]) > 0:
            env["cuts"][-1]["space"] = len(sols)
        # TODO check whether all are still in table

        for row in T:
            assert row.tolist() in sols, f"Removed sol: {row}"

        if env["debug_unlucky"]:
            # force getting unlucky
            non_sol = next((sol for sol in sols if sol not in T.tolist()), sols[0])
            for x, a in zip(X, non_sol):
                x._value = a
        else:
            X.clear()

        assert len(sols)

    log("Model:", model, env=env, verbosity=2)
    log("Solving.. ", env=env, end="")
    slv = cp.SolverLookup.get(env.get("solver", "gurobi"), model)
    slv.native_model.Params.LazyConstraints = 1
    # slv.native_model.Params.LogFile = "/tmp/gurobi.log"
    if env["verbosity"] >= 3:
        slv.native_model.Params.OutputFlag = 1

    # while True with a counter
    for iteration in itertools.count(start=1):
        # https://or.stackexchange.com/questions/12591/ensure-gurobi-uses-callback-on-all-feasible-solutions It looks like if the solution at the end of the root node is integer, gurobi doesn't pass through callbacks for fractional solutions.
        hassol = slv.solve(solution_callback=get_solution_callback(slv, ivarmap, T_enc, env))

        for enc in ivarmap.values():
            enc._x._value = enc.decode()

        assert hassol, f"Unsat model for {slv}"

        if table.value():
            return True

        A_enc = [x.value() for x in X_enc]
        explanation = explain(A_enc, T_enc, env)
        log_explanation(explanation, "SOL", env)
        assert explanation, "Expected explanation for integer solution"
        explanation = cp.sum(X_enc[c] for c in explanation) < len(explanation)
        assert explanation is not False, f"{A_enc}"
        log(f"  constraint == {explanation}", env=env)
        slv += [explanation]

        if iteration > env.get("max_iterations", iteration):
            raise Exception(f"Out of iterations ({env['max_iterations']=})")
            return False


# TODO use gurobi lazy constraints interface


def show_env(env):
    log(", ".join(f"{k}={env[k]}" for k in ["shrink", "heuristic"]), env=env, verbosity=0)
    # log(f"cuts = {env['cuts']}", env=env, verbosity=1)
    log(f"n_cuts = {sum(len(c['cut']) > 0 for c in env['cuts'])}", env=env, verbosity=0)
    log(f"n_unexplained = {list(len(c['cut']) for c in env['cuts']).count(0)}", env=env, verbosity=0)
    log(
        "cut cardinalities/strengths:",
        ", ".join(f"{len(c['cut'])}" for c in env["cuts"]),
        env=env,
        verbosity=2,
    )


def main():
    envs = [
        {
            "solver": "gurobi",
            "shrink": shrink,
            "heuristic": heuristic,
            "cuts": [],
            "verbosity": 1,
        }
        for heuristic in [
            # heuristics
            Heuristic.INPUT,
            Heuristic.GREEDY,
            Heuristic.REDUCE,
        ]
        for shrink in [
            # shrink
            # False,
            True,
        ]
    ]

    # STATS = {
    #         "explanations":
    #         }

    # X, T = generate_table_from_data([[1, 1], [2, 2]], 3)
    # X, T = generate_table_from_example()
    # X, T = generate_table(2, 2, 3)
    # X, T = generate_table(3, 10, 5)

    # X, T = generate_table(5, 25, 10)
    X, T = generate_table(10, 100, 10)

    # envs = envs[0:1]
    for env in envs:
        assert solve(X, T, env)
        A = [x.value() for x in X]
        assert A in T.tolist(), f"Check failed: assignment {A} was not in the table."
        X.clear()

    log("STATS", env=env)
    for env in envs:
        show_env(env)


if __name__ == "__main__":
    random.seed(42)
    assert pytest.main() == pytest.ExitCode.OK
    main()


@pytest.fixture()
def env():
    yield {"verbosity": 4}


class TestTables:
    def test_explain_frac(self, env):
        X, T = generate_table_from_example()
        T_enc = encode(X, T)
        assert (  # Example 1 from assignment [2,2,2]
            explain([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], T_enc, env) == {1, 5}
        )
        assert (  # Example 6
            explain([0.0, 0.5, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], T_enc, env) == set()
        )

    @pytest.mark.parametrize(
        "table",
        (
            generate_table_from_data([[1, 1], [2, 2]], 3),
            generate_table_from_example(),
            generate_table(2, 2, 3),
            generate_table(3, 10, 5),
            generate_table(5, 25, 10),
            # generate_table(10, 100, 10),
        ),
    )
    def test_tables(self, table, env):
        X, T = table
        assert solve(X, T, env), "Expected feasible"
        A = [x.value() for x in X]
        assert A in T.tolist(), f"Check failed: assignment {A} was not in the table."
