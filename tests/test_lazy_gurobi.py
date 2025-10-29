import numpy as np
import math

import pytest
import cpmpy as cp


from cpmpy.solvers.lazy_gurobi import CPM_lazy_gurobi, Heuristic, encode
from cpmpy.tools.xcsp3 import XCSP3Dataset, read_xcsp3

import random


def generate_table_from_example():
    x = cp.intvar(1, 4, name="x")
    y = cp.intvar(1, 3, name="y")
    z = cp.intvar(1, 3, name="z")
    X = (x, y, z)
    T = np.array([(2, 1, 1), (3, 2, 2), (4, 3, 3), (1, 2, 3), (2, 1, 2)])
    return cp.Model(cp.Table(X, T))


def generate_two_tables():
    x = cp.intvar(1, 4, name="x")
    y = cp.intvar(1, 3, name="y")
    z = cp.intvar(2, 4, name="z")
    return cp.Model(
        cp.Table((x, y, z), np.array([(2, 1, 2), (3, 2, 2), (4, 3, 3), (1, 2, 3), (2, 1, 4)])),
        cp.Table((z, y), np.array([(2, 1), (3, 2), (2, 3), (2, 3), (4, 1)])),
    )


def generate_table_from_data(T, d):
    """Generate a table constraint with the given `rows` and with var domains of size `d`"""
    X = cp.intvar(1, d, shape=len(T[0]), name="x")
    return cp.Model(cp.Table(X, np.array(T)))


def generate_table(n, m, d, k=1):
    """Generate a table constraint with `n` variables with domains of size `d`, and with `m` rows"""
    X = cp.intvar(1, d, shape=n, name="x")
    # model = cp.Model(x == x for x in X)
    model = cp.Model()
    for _ in range(k):
        Y = random.choices(X, k=n // 2)
        # Y = X
        if len(Y):
            T = np.array([tuple(random.randint(1, d) for _ in enumerate(Y)) for _ in range(m)])
            model += cp.Table(Y, T)
    return model


def assert_integer_solution(A_enc):
    for a_enc_i in A_enc:
        assert math.isclose(a_enc_i, round(a_enc_i), abs_tol=1e-5), (
            f"Expected integer solution for MIP, but got {a_enc_i} in {A_enc}"
        )


def show_assignment(X):
    return ", ".join(f"{x}={x.value()}" for x in X)


def check_model(model):
    violations = [c for c in model.constraints if c.value() is False]
    X = cp.transformations.get_variables.get_variables_model(model)
    if model.copy().solve():
        assert all(x.value() is not None for x in X), (
            f"Expected all variables to be assigned, but found: {show_assignment(X)}"
        )
        assert not violations, (
            f"Infeasible constraints for assignment:\n\n{show_assignment(X)}\n\n{'\n\n'.join(str(v) for v in violations)}"
        )
    else:
        assert all(x.value() is None for x in X), (
            f"Expected all variables to be assigned, but found: {show_assignment(X)}"
        )


def show_sols(sols, T):
    return ", ".join(f"*{sol}" if list(sol) in T.tolist() else f"{sol}" for sol in sorted(sols))


def with_constraints(model, with_alldiff=False, with_min=False):
    X = cp.transformations.get_variables.get_variables_model(model)
    if with_alldiff:
        model += cp.AllDifferent(X)
    if with_min:
        model.minimize(sum(X))
    return model


@pytest.fixture()
def env():
    yield {"verbosity": 2}


class TestTables:
    def test_explain_frac(self, env):
        slv = CPM_lazy_gurobi(env=env)
        X, T = generate_table_from_example().constraints[0].args
        T_enc = encode(X, T)
        assert (  # Example 1 from assignment [2,2,2]
            slv.explain([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], T_enc) == {1, 5}
        )
        assert (  # Example 6
            slv.explain([0.0, 0.5, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], T_enc) == set()
        )

    @pytest.mark.parametrize(
        "model",
        (
            cp.Model(cp.AllDifferent(cp.intvar(1, 3, shape=3))),
            generate_table_from_data([(1, 1), (2, 2)], 3),  # Feasible
            with_constraints(generate_table_from_data([(1, 1), (2, 2)], 3), with_alldiff=True),  # Infeasible
            with_constraints(generate_table_from_data([(1, 2), (2, 1)], 3), with_alldiff=True, with_min=True),
            generate_table_from_example(),
            generate_two_tables(),
            with_constraints(generate_table(2, 2, 3), with_alldiff=True, with_min=True),
            with_constraints(generate_table(3, 10, 5), with_alldiff=True, with_min=True),
            with_constraints(generate_table(6, 10, 10), with_alldiff=True, with_min=True),
            with_constraints(generate_table(2, 2, 3, k=2)),
        ),
    )
    def test_models(self, model, env):
        print("Test model:")
        print(model)
        is_sat = model.solve()
        assert CPM_lazy_gurobi(model, env=env).solve() == is_sat, "Expected equisat"
        check_model(model)

    def test_table_enc(self, env):
        x = cp.intvar(1, 4, name="x")
        y = cp.intvar(1, 3, name="y")
        z = cp.intvar(1, 3, name="z")
        X = (x, y, z)
        T = np.array([(2, 1, 1), (3, 2, 2), (4, 3, 3), (1, 2, 3), (2, 1, 2)])
        model = cp.Model(cp.Table(X, T), cp.AllDifferent(X))
        is_sat = model.solve()
        assert CPM_lazy_gurobi(model, env=env).solve() == is_sat, "Expected equisat"
        print("Test model", model)
        print("TF", CPM_lazy_gurobi(model, env=env).transform(model.constraints))
        check_model(model)

