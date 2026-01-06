import math
import random

import numpy as np
import pytest

import cpmpy as cp
from cpmpy.solvers.lazy_gurobi import CPM_lazy_gurobi


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


def generate_table(n, m, d, k=1, allow_duplicate_vars=False):
    """Generate a table constraint with `n` variables with domains of size `d`, and with `m` rows"""
    X = cp.intvar(1, d, shape=n, name="x")
    # model = cp.Model(x == x for x in X)
    model = cp.Model()
    random.seed(SEED)
    for _ in range(k):
        k = n // 2
        X = list(X)
        Y = random.sample(X, k=k) if allow_duplicate_vars else random.choices(X, k=k)
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


def check_model(model, env=None):
    expected_sat = model.deepcopy().solve()

    slv = CPM_lazy_gurobi(cpm_model=model, env=env)
    actual_sat = slv.solve()
    slv.stats()
    assert expected_sat == actual_sat, "Expected equisat"

    if expected_sat:
        X = cp.transformations.get_variables.get_variables_model(model)
        assert all(x.value() is not None for x in X), (
            f"Expected all variables to be assigned, but found: {show_assignment(X)}"
        )

        violations = [c for c in model.constraints if c.value() is False]
        assert not violations, f"""For assignment:

{show_assignment(X)}

The following constraints are Infeasible:

{"\n\n".join(str(v) for v in violations)}
        """
    print("PASS.")


def show_sols(sols, T):
    return ", ".join(f"*{sol}" if list(sol) in T.tolist() else f"{sol}" for sol in sorted(sols))


def with_constraints(model, with_alldiff=False, with_min=False):
    X = cp.transformations.get_variables.get_variables_model(model)
    if with_alldiff:
        model += cp.AllDifferent(X)
    if with_min:
        model.minimize(sum(X))
    return model


SEED = 42
SEED = None


@pytest.fixture()
def env():
    yield {"verbosity": 2, "debug": False, "max_iterations": 1000, "seed": SEED}


class TestTables:
    def test_explain(self, env):
        slv = CPM_lazy_gurobi(
            cpm_model=cp.Model(generate_table_from_example().constraints),
            env={**env, **{"shrink": False, "debug": True}},
        )
        # X, T = generate_table_from_example().constraints[0].args
        slv += generate_table_from_example().constraints
        # slv.add(generate_table_from_example().constraints)
        X_enc, T_enc, parts, table = slv.tables[0]

        assert parts == [4, 3, 3]

        # T_enc = encode(X, T)
        # parts = [4, 3, 3]
        A_enc = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]

        def list_to_A_enc(A_enc):
            return dict(zip(X_enc, A_enc))

        A_enc = list_to_A_enc(A_enc)

        explanations = slv._explain_assignment(A_enc)
        print("E", list(explanations))
        # explanation = slv.explain(A_enc, T_enc, parts)
        # assert (  # Example 1 from assignment [2,2,2]
        #     explanation == {1, 5}
        # )

        with pytest.raises(AssertionError) as e:
            slv.check_explanation(cp.all(X_enc), X_enc, A_enc, T_enc, table)
        print("ERR", e.value)

        # slv.check_explanation(explanation, X_enc, A_enc, T_enc)
        # assert (  # Example 7; no longer in use since explain_frac2
        #     slv.explain([0.0, 0.5, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], T_enc) == {1, 5, 8}
        # )

        # assert (  # Example 9
        #     slv.explain([0.0, 0.5, 0.5, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.0], T_enc, parts) == {1, 5}
        # )

    @pytest.mark.parametrize(
        "case",
        (
            (i, j, t)
            for j in range(1000)  # to repeat the test
            for i, t in enumerate(
                (
                    cp.Model(cp.AllDifferent(cp.intvar(1, 3, shape=3))),
                    generate_table_from_data([(1, 1), (2, 2)], 3),  # Feasible
                    with_constraints(
                        generate_table_from_data([(1, 1), (2, 2)], 3), with_alldiff=True
                    ),  # Infeasible
                    with_constraints(
                        generate_table_from_data([(1, 2), (2, 1)], 3), with_alldiff=True, with_min=True
                    ),
                    generate_table_from_example(),
                    generate_two_tables(),
                    with_constraints(generate_table(2, 2, 3), with_alldiff=True, with_min=True),
                    with_constraints(generate_table(4, 4, 4), with_alldiff=False, with_min=False),
                    with_constraints(generate_table(6, 10, 5), with_alldiff=False, with_min=True),
                    with_constraints(generate_table(2, 2, 3, k=2)),
                    with_constraints(generate_table(6, 6, 4, k=3)),
                    # with_constraints(generate_table(6, 4, 4)),  # minimized 1/1000 bug
                    # cp.Model(
                    #     cp.Table(
                    #         [cp.intvar(1, 4), cp.intvar(1, 4), cp.intvar(1, 4)],
                    #         [[4, 4, 2], [3, 3, 3], [3, 2, 2], [1, 4, 4]],
                    #     )  # 1/1000 bug
                    # ),
                )
            )
        ),
        ids=lambda val: val[0],
    )
    def test_models(self, case, env):
        _, _, model = case
        print("Test model:")
        print(model)
        check_model(model, env=env)

    def test_table_enc(self, env):
        x = cp.intvar(1, 4, name="x")
        y = cp.intvar(1, 3, name="y")
        z = cp.intvar(1, 3, name="z")
        X = (x, y, z)
        T = np.array([(2, 1, 1), (3, 2, 2), (4, 3, 3), (1, 2, 3), (2, 1, 2)])
        model = cp.Model(cp.Table(X, T), cp.AllDifferent(X))
        print("Test model", model)
        # print("TF", CPM_lazy_gurobi(cpm_model=model, env=env).transform(model.constraints))
        check_model(model)
