#!/usr/bin/env python
import numpy as np
import cpmpy as cp

import random

VERBOSITY = 1
DEBUG = True
LOOP_LIMIT = None
random.seed(42)


def log(*mess, verbosity=1):
    if verbosity <= VERBOSITY:
        print(*mess)


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


X, T = generate_table_from_data([[1, 1], [2, 2]], 3)
# X, T = generate_table(2, 2, 3)
# X, T = generate_table(3, 10, 5)
# X, T = generate_table(10, 100, 10)
log(X, verbosity=2)
log(T, verbosity=2)

# TODO how to determine it is a failure and not a solution?


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


T_ = encode(X, T)
log(T_, verbosity=2)


def rows(T, i, j=1):
    """Row indices where T[i]==j"""
    return set(int(i) for i in np.where(T[:, i] == j)[0].flatten())


def cols(T, i, j=1):
    """Col indices where T[i]==j"""
    return set(int(i) for i in np.where(T[i, :] == j)[0].flatten())


def explain(A, T, X):
    log(f"Explain {A}")
    C = set()  # vars added to cut
    R = set(range(len(T)))
    dbg = 0
    while len(R):
        i = min((i for i in (A - C)), default=None)  # choose some col which is 1
        if i is None:  # TODO [?] slightly different stopping condition
            break
        log("A", A, C, verbosity=3)
        log(f"adding {i + 1}", verbosity=3)
        T_i = rows(T, i)
        R = R.intersection(T_i)
        C.add(i)
        dbg += 1
        assert LOOP_LIMIT is None or dbg < LOOP_LIMIT

    constraint = cp.any(~X[i] for i in C)
    log(f"  by explanation ({len(C)}): {C} == {constraint}")
    return constraint


# assert rows(T, 0, 2) == {0, 4}
# assert rows(T_, 1) == {0, 4}


def show_sols(sols, T):
    return ", ".join(f"*{sol}" if list(sol) in T.tolist() else f"{sol}" for sol in sorted(sols))


REAL = True
if REAL:
    X_ = [x == d for x in X for d in range(x.lb, x.ub + 1)]
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
        log("A", A, constraint.value())
        # A = [x.value() for x in X]

        if constraint.value():
            break

        A = [encode_x(a, x.lb, x.ub) for a, x in zip(A, X)]
        A = [ai for a in A for ai in a]
        # A = set(np.where(np.array(A) == 1)[0])
        A = cols(np.array([A]), 0)

        explanation = explain(A, T_, X_)
        m += [explanation]
        log("M", m, verbosity=2)

        dbg += 1
        assert LOOP_LIMIT is None or dbg < LOOP_LIMIT
    print("Satisfying sol:", A)

    A = [x.value() for x in X]
else:
    A = [2, 2, 2]

# TODO take care of encoding to integer
# TODO use native model to resolve and

# m.solve(solver="gurobi")
