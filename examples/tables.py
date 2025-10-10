#!/usr/bin/env python
import pytest
import numpy as np
import math
import cpmpy as cp

from cpmpy.solvers.lazy_gurobi import CPM_lazy_gurobi, Heuristic
from cpmpy.tools.xcsp3 import XCSP3Dataset, read_xcsp3

import random


def load_xcsp():
    tables = []
    seen_problems = set()
    for year, track in (
        (2025, "COP25"),
        # (2025, "MiniCOP25"),
        # (2024, "COP"),
    ):
        max_iterations = None
        for i, (filename, metadata) in enumerate(XCSP3Dataset(year=year, track=track, download=True)):
            # Do whatever you want here, e.g. reading to a CPMpy model and solving it:
            print("f", filename, metadata)

            def get_problem_name(name):
                return name.split("-")[0]

            problem_name = get_problem_name(metadata["name"])
            if problem_name in seen_problems:
                continue
            else:
                seen_problems.add(problem_name)

            model = read_xcsp3(filename)
            if model is None:
                continue

            instance_tables = []
            for c in model.constraints:
                if isinstance(c, cp.expressions.core.Expression) and c.name == "table":
                    _, tab = c.args
                    height, width = len(tab), len(tab[0])
                    instance_tables.append((height, width))

            tables.append((filename, sum(w * h for w, h in instance_tables), instance_tables))

            if max_iterations is not None and i > max_iterations:
                break

    print("TABLES", tables)
    with open("./cpmpy/tools/xcsp3/tables.txt", "w") as file:
        file.write("\n".join(":".join(str(t) for t in table) for table in tables))
    return


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

    # model = generate_table_from_data([[1, 1], [2, 2]], 3)
    # model = generate_table_from_example()
    # model = generate_table(2, 2, 3)
    # model = generate_table(3, 10, 5)
    # model = generate_table(5, 25, 10)
    model = generate_table(10, 100, 10)

    X = cp.transformations.get_variables.get_variables_model(model)
    model += cp.AllDifferent(X)
    is_sat = model.copy().solve()

    # envs = envs[0:1]
    for env in envs:
        slv = CPM_lazy_gurobi(model.copy(), env=env)
        assert slv.solve() == is_sat
        print("SOL", X)
        print(show_assignment(X))
        slv.show_env()
        check_model(model)
        X.clear()

    print("STATS")
    for env in envs:
        CPM_lazy_gurobi(env=env).show_env()


if __name__ == "__main__":
    random.seed(42)
    # assert pytest.main() == pytest.ExitCode.OK
    main()


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
            generate_table_from_data([[1, 1], [2, 2]], 3),  # Infeasible w/ AllDiff
            with_alldiff(generate_table_from_data([[1, 2], [2, 1]], 3), with_alldiff=True, with_min=True),
            generate_table_from_example(),
            with_alldiff(generate_table(2, 2, 3), with_alldiff=True, with_min=True),
            with_alldiff(generate_table(3, 10, 5), with_alldiff=True, with_min=True),
            with_alldiff(generate_table(6, 10, 10), with_alldiff=True, with_min=True),
        ),
    )
    def test_models(self, model, env):
        is_sat = model.solve()
        assert CPM_lazy_gurobi(model, env=env).solve() == is_sat, "Expected equisat"
        check_model(model)

    #
    # if env.get("debug", False):
    #     n_sols = model.solveAll(
    #         display=lambda: sols.append([x.value() for x in X]),
    #         solver=env.get("solver", "gurobi"),
    #         solution_limit=1000 if env["solver"] == "gurobi" else None,
    #     )
    #     assert env["solver"] != "gurobi" or n_sols < 1000
    #
    #     self.log(f"Search space remaining: ({len(sols)})")
    #     self.log(show_sols(sols, T), verbosity=2)
    #     if len(env["cuts"]) > 0:
    #         env["cuts"][-1]["space"] = len(sols)
    #     # TODO check whether all are still in table
    #
    #     for row in T:
    #         assert row.tolist() in sols, f"Removed sol: {row}"
    #
    #     if env["debug_unlucky"]:
    #         # force getting unlucky
    #         non_sol = next((sol for sol in sols if sol not in T.tolist()), sols[0])
    #         for x, a in zip(X, non_sol):
    #             x._value = a
    #     else:
    #         X.clear()
    #
    #     assert len(sols)
