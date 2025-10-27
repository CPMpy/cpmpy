#!/usr/bin/env python
import numpy as np
import cpmpy as cp

import random


def log(*mess, verbosity=1, end="\n"):
    if verbosity <= VERBOSITY:
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


def shrink(C, T):
    for i in C:
        log(f"shrinking {i} in {C}", verbosity=3)

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


def choose(A, T_enc, R, heuristic=Heuristic.INPUT):
    if len(A) == 0:
        return None
    match heuristic:
        case Heuristic.INPUT:
            return min(i for i in A)
        case Heuristic.GREEDY:
            return min(A, key=lambda i: len(R.intersection(rows(T_enc, i))))
        case Heuristic.REDUCE:
            return min(
                (i for i in A if R.intersection(rows(T_enc, i)) != R), key=lambda i: len(rows(T_enc, i))
            )


def explain(A_enc, T_enc, X_enc, env):
    log(f"Explain {A_enc} {[X_enc[i] for i in A_enc]}")
    C = set()  # vars added to cut
    R = set(range(len(T_enc)))
    dbg = 0
    while len(R):
        # choose some col which is 1
        i = choose(A_enc - C, T_enc, R, heuristic=env["heuristic"])

        if i is None:  # TODO [?] slightly different stopping condition
            break

        log("A", A_enc, C, verbosity=3)
        log(f"adding {i + 1}", verbosity=3)
        T_i = rows(T_enc, i)
        R = R.intersection(T_i)
        C.add(i)
        dbg += 1
        assert LOOP_LIMIT is None or dbg < LOOP_LIMIT

    log(f"  by explanation of size ({len(C)}): {C}")
    if env["shrink"]:
        size = len(C)
        C = shrink(C, T_enc)
        if len(C) < size:
            log(f"  shrunk to size ({len(C)}): {C}")
            # assert False, "TODO; shrinking is not triggering"

    if env["cuts"] is not None:
        assert C not in env["cuts"], f"Already found cut {C} previously in {env['cuts']}"
        env["cuts"].append({"cut": C, "shrunk": 0})

    constraint = cp.any(~X_enc[i] for i in C)
    log(f"  cut == {constraint}")
    return constraint


# assert rows(T, 0, 2) == {0, 4}
# assert rows(T_, 1) == {0, 4}


def show_sols(sols, T):
    return ", ".join(f"*{sol}" if list(sol) in T.tolist() else f"{sol}" for sol in sorted(sols))


def solve(X, T, env):
    log("X =", ", ".join(f"{x} in {x.lb}..{x.ub}" for x in X), verbosity=0)
    log("T =", verbosity=1)
    log(T, verbosity=1)
    T_enc = encode(X, T)
    log("T_enc =", verbosity=1)
    log(T_enc, verbosity=1)

    X_enc = [x == d for x in X for d in range(x.lb, x.ub + 1)]
    m = cp.Model([cp.sum(x == d for d in range(x.lb, x.ub + 1)) == 1 for x in X])
    constraint = cp.Table(X, T)

    dbg = 0
    while True:
        sols = []

        if DEBUG:
            n_sols = m.solveAll(
                display=lambda: sols.append([x.value() for x in X]),
                solver=env["solver"],
                solution_limit=1000 if env["solver"] == "gurobi" else None,
            )
            assert env["solver"] != "gurobi" or n_sols < 1000

            log(f"Search space remaining: ({len(sols)})")
            log(show_sols(sols, T), verbosity=2)
            if len(env["cuts"]) > 0:
                env["cuts"][-1]["space"] = len(sols)
            # TODO check whether all are still in table

            for row in T:
                assert row.tolist() in sols, f"Removed sol: {row}"

            if DEBUG_UNLUCKY:
                # force getting unlucky
                non_sol = next((sol for sol in sols if sol not in T.tolist()), sols[0])
                for x, a in zip(X, non_sol):
                    x._value = a
            else:
                X.clear()

            assert len(sols)

        if any(x._value is None for x in X):
            log("Solving.. ", end="")
            assert m.solve(solver=env["solver"])

        A = [x.value() for x in X]
        log(A)

        if constraint.value():
            return A

        X.clear()

        # encode assignment
        A_enc = [encode_x(a, x.lb, x.ub) for a, x in zip(A, X)]
        A_enc = [ai for a in A_enc for ai in a]
        A_enc = cols(np.array([A_enc]), 0)

        explanation = explain(A_enc, T_enc, X_enc, env)
        m += [explanation]
        log("New model", verbosity=3)
        log(m, verbosity=3)

        dbg += 1
        assert LOOP_LIMIT is None or dbg < LOOP_LIMIT


# TODO use gurobi lazy constraints interface


def show_env(env):
    log(", ".join(f"{k}={env[k]}" for k in ["shrink", "heuristic"]), verbosity=0)
    log(f"n_cuts = {len(env['cuts'])}", verbosity=0)
    log(
        "cut cardinalities/strengths:",
        ", ".join(f"{len(c['cut'])} / {c.get('space', '?')}" for c in env["cuts"]),
        verbosity=2,
    )


if __name__ == "__main__":
    VERBOSITY = 1
    DEBUG = True
    DEBUG_UNLUCKY = False
    LOOP_LIMIT = None
    random.seed(42)

    envs = [
        {
            "solver": "gurobi",
            # "solver": "ortools",
            "shrink": shrink,
            "heuristic": heuristic,
            "cuts": [],
        }
        for heuristic in [
            # heuristics
            Heuristic.INPUT,
            # Heuristic.GREEDY,
            # Heuristic.REDUCE,
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

    X, T = generate_table_from_data([[1, 1], [2, 2]], 3)
    # X, T = generate_table_from_example()
    # X, T = generate_table(2, 2, 3)
    # X, T = generate_table(3, 10, 5)

    # DEBUG = False
    # X, T = generate_table(5, 25, 10)
    # X, T = generate_table(10, 100, 10)

    # envs = envs[0:1]
    for env in envs:
        A = solve(X, T, env)
        assert A in T.tolist(), "Check failed."
        X.clear()

    log("STATS")
    for env in envs:
        log(", ".join(f"{k}={env[k]}" for k in ["shrink", "heuristic"]), verbosity=0)
        log(f"n_cuts = {len(env['cuts'])}", verbosity=0)
        log(
            "cut cardinalities/strengths:",
            ", ".join(f"{len(c['cut'])} / {c.get('space', '?')}" for c in env["cuts"]),
            verbosity=2,
        )
