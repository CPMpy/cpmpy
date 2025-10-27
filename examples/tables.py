#!/usr/bin/env python
import numpy as np
import cpmpy as cp

import random


def log(*mess, verbosity=1, end="\n"):
    if verbosity <= VERBOSITY:
        print(*mess, end=end)


# x = cp.intvar(1, 4, name="x")
# y = cp.intvar(1, 3, name="y")
# z = cp.intvar(1, 3, name="z")
# X = (x, y, z)
# T = [(2, 1, 1), (3, 2, 2), (4, 3, 3), (1, 2, 3), (2, 1, 2)]
# T = np.array(T)


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
        print(f"shrinking {i}")
        S = set.intersection({rows(T, l) for l in C if l != i})
        print(f"intersection: ", S)
        if len(S) == 0:
            C = C - {i}
    return C


def explain(A_enc, T_enc, X_enc):
    log(f"Explain {A_enc} {[X_enc[i] for i in A_enc]}")
    C = set()  # vars added to cut
    R = set(range(len(T_enc)))
    dbg = 0
    while len(R):
        i = min((i for i in (A_enc - C)), default=None)  # choose some col which is 1
        if i is None:  # TODO [?] slightly different stopping condition
            break
        log("A", A_enc, C, verbosity=3)
        log(f"adding {i + 1}", verbosity=3)
        T_i = rows(T_enc, i)
        R = R.intersection(T_i)
        C.add(i)
        dbg += 1
        assert LOOP_LIMIT is None or dbg < LOOP_LIMIT

    log(f"  by explanation ({len(C)}): {C}")
    if SHRINK:
        C = shrink(C, T_enc)
        log(f"  shrunk to ({len(C)}): {C}")
    # sum X[i] < len(X)
    constraint = cp.any(~X_enc[i] for i in C)
    log(f"  cut == {constraint}")
    return constraint


# assert rows(T, 0, 2) == {0, 4}
# assert rows(T_, 1) == {0, 4}


def show_sols(sols, T):
    return ", ".join(f"*{sol}" if list(sol) in T.tolist() else f"{sol}" for sol in sorted(sols))


def solve(X, T):
    log("X =", ", ".join(f"{x} in {x.lb}..{x.ub}" for x in X), verbosity=2)
    log("T =", verbosity=2)
    log(T, verbosity=2)
    T_enc = encode(X, T)
    log("T_enc =", verbosity=2)
    log(T_enc, verbosity=2)

    X_enc = [x == d for x in X for d in range(x.lb, x.ub + 1)]
    m = cp.Model([cp.sum(x == d for d in range(x.lb, x.ub + 1)) == 1 for x in X])
    constraint = cp.Table(X, T)

    dbg = 0
    while True:
        sols = []

        if DEBUG:
            m.solveAll(display=lambda: sols.append([x.value() for x in X]))
            log(f"Search space remaining: ({len(sols)})")
            log(show_sols(sols, T), verbosity=2)
            # TODO check whether all are still in table

            for t in T:
                assert t.tolist() in sols, f"Removed sol: {t}"

            assert len(sols)
        # assert m.solve()

        m.solve()
        A = [x.value() for x in X]

        if constraint.value():
            log("Satisfying sol:", A)
            return A

        A_enc = [encode_x(a, x.lb, x.ub) for a, x in zip(A, X)]
        A_enc = [ai for a in A_enc for ai in a]
        # A = set(np.where(np.array(A) == 1)[0])
        A_enc = cols(np.array([A_enc]), 0)

        explanation = explain(A_enc, T_enc, X_enc)
        m += [explanation]
        log("New model\n", m, verbosity=3)

        dbg += 1
        assert LOOP_LIMIT is None or dbg < LOOP_LIMIT


# TODO use gurobi lazy constraints interface

if __name__ == "__main__":
    VERBOSITY = 2
    DEBUG = True
    LOOP_LIMIT = None
    SHRINK = False
    random.seed(42)

    X, T = generate_table_from_data([[1, 1], [2, 2]], 3)
    # X, T = generate_table(2, 2, 3)
    # X, T = generate_table(3, 10, 5)
    # X, T = generate_table(10, 100, 10)

    solve(X, T)
