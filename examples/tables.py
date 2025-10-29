#!/usr/bin/env python
import pytest
import numpy as np
import math
import cpmpy as cp
from pathlib import Path

from cpmpy.solvers.lazy_gurobi import CPM_lazy_gurobi, Heuristic, CPM_gurobi
from cpmpy.tools.xcsp3 import XCSP3Dataset, read_xcsp3

import random


def generate_table_from_example():
    x = cp.intvar(1, 4, name="x")
    y = cp.intvar(1, 3, name="y")
    z = cp.intvar(1, 3, name="z")
    X = (x, y, z)
    T = np.array([(2, 1, 1), (3, 2, 2), (4, 3, 3), (1, 2, 3), (2, 1, 2)])
    return cp.Model(cp.Table(X, T))


def run_xcsp3_table_instances(env):
    tables = []
    seen_problems = set()
    instances = Path("./cpmpy/tools/xcsp3/tables.txt")

    if True or not instances.exists():
        for year, track in (
            (2025, "COP25"),
            # (2025, "MiniCOP25"),
            # (2024, "COP"),
        ):
            max_iterations = None
            for i, (filename, metadata) in enumerate(XCSP3Dataset(year=year, track=track, download=True)):
                # Do whatever you want here, e.g. reading to a CPMpy model and solving it:

                def get_problem_name(name):
                    return name.split("-")[0]

                problem_name = get_problem_name(metadata["name"])

                if problem_name in seen_problems and False:
                    continue
                else:
                    seen_problems.add(problem_name)

                if True:
                    # target = "SchedulingOS"
                    target = "AlteredStates"
                    # target = "FAPP"
                    target = "TankAllocation2"
                    if problem_name == target:
                        print(problem_name)

                        time_limit = 60
                        model = read_xcsp3(filename)

                        # if True:
                        if False:
                            slv = CPM_gurobi(cpm_model=model.copy())
                            slv.solve(time_limit=time_limit)
                            print(slv.status(), slv.objective_value())

                        slv = CPM_lazy_gurobi(cpm_model=model.copy(), env=env)
                        slv.solve(time_limit=time_limit)
                        print(slv.status(), slv.objective_value())
                        slv.stats()

                        return
                    else:
                        continue

                model = read_xcsp3(filename)
                if model is None:
                    continue

                instance_tables = []
                for c in model.constraints:
                    if isinstance(c, cp.expressions.core.Expression) and c.name == "table":
                        # TODO involve int var doms
                        _, tab = c.args
                        rows, cols = len(tab), len(tab[0])
                        instance_tables.append((rows, cols))

                tables.append((filename, sum(w * h for w, h in instance_tables), instance_tables))

                if max_iterations is not None and i > max_iterations:
                    break

        print("TABLES", tables)
        with open(instances, "w") as file:
            file.write("\n".join(":".join(str(t) for t in table) for table in tables))

    with open(instances) as instances:
        for f in instances.read().splitlines():
            print("F", f)

    return


def main():
    envs = [
        {
            "solver": "gurobi",
            "shrink": shrink,
            "heuristic": heuristic,
            "explain_fractional": True,
            "cuts": [],
            "verbosity": 1,
        }
        for heuristic in [
            # heuristics
            # Heuristic.INPUT,
            # Heuristic.GREEDY,
            Heuristic.REDUCE,
        ]
        for shrink in [
            # shrink
            # False,
            True,
        ]
    ]

    for env in envs:
        run_xcsp3_table_instances(env)
    return
    model = generate_table_from_example()

    X = cp.transformations.get_variables.get_variables_model(model)
    model += cp.AllDifferent(X)
    is_sat = model.copy().solve()

    # envs = envs[0:1]
    for env in envs:
        slv = CPM_lazy_gurobi(model.copy(), env=env)
        assert slv.solve() == is_sat
        print("SOL", X)
        # print(show_assignment(X))
        # slv.show_env()
        # check_model(model)
        # X.clear()

    print("STATS")
    for env in envs:
        CPM_lazy_gurobi(env=env).stats()


if __name__ == "__main__":
    random.seed(42)
    # assert pytest.main() == pytest.ExitCode.OK
    main()
