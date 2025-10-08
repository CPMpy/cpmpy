#!/usr/bin/env python
import sys
import gurobipy as gp
from gurobipy import GRB
import pytest
import itertools
import numpy as np
import math
import cpmpy as cp

from cpmpy.tools.xcsp3 import XCSP3Dataset, read_xcsp3

import random


def show_assignment(X):
    return ", ".join(f"{x}={x.value()}" for x in X)


def check_model(model):
    violations = [c for c in model.constraints if not c.value()]
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


def with_alldiff(model, with_alldiff=False, with_min=False):
    X = cp.transformations.get_variables.get_variables_model(model)
    if with_alldiff:
        model += cp.AllDifferent(X)
    if with_min:
        model.minimize(sum(X))
    return model


def generate_table_from_example():
    x = cp.intvar(1, 4, name="x")
    y = cp.intvar(1, 3, name="y")
    z = cp.intvar(1, 3, name="z")
    X = (x, y, z)
    T = [(2, 1, 1), (3, 2, 2), (4, 3, 3), (1, 2, 3), (2, 1, 2)]
    T = np.array(T)
    return cp.Model(cp.Table(X, T))


def generate_table_from_data(rows, d):
    """Generate a table constraint with the given `rows` and with var domains of size `d`"""
    X = cp.intvar(1, d, shape=len(rows[0]), name="x")
    T = np.array(rows)
    return cp.Model(cp.Table(X, T))


def generate_table(n, m, d):
    """Generate a table constraint with `n` variables with domains of size `d`, and with `m` rows"""
    X = cp.intvar(1, d, shape=n, name="x")
    T = np.array([tuple(random.randint(1, d) for _ in range(n)) for _ in range(m)])
    model = cp.Model(cp.Table(X, T))
    if with_alldiff:
        model += cp.AllDifferent(X)
    return model


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


from enum import Enum


class Heuristic(Enum):
    INPUT = 1
    GREEDY = 2
    REDUCE = 3


def show_sols(sols, T):
    return ", ".join(f"*{sol}" if list(sol) in T.tolist() else f"{sol}" for sol in sorted(sols))


def assert_integer_solution(A_enc):
    for a_enc_i in A_enc:
        assert math.isclose(a_enc_i, round(a_enc_i), abs_tol=1e-5), (
            f"Expected integer solution for MIP, but got {a_enc_i} in {A_enc}"
        )


def is_integer_solution(A_enc):
    return all(math.isclose(a_enc_i, a_enc_i > 0.5, abs_tol=1e-5) for a_enc_i in A_enc)


def get_x_enc(ivarmap):
    return [x_enc_i for x_enc in ivarmap.values() for x_enc_i in x_enc._xs]


class CPM_lazy_gurobi(cp.solvers.gurobi.CPM_gurobi):
    def __init__(self, cpm_model=None, subsolver=None, env=None):
        self.env = {}
        self.env["debug"] = False
        self.env["debug_unlucky"] = False  # TODO re-enable
        self.env["cuts"] = []
        self.env["verbosity"] = 1
        self.env["heuristic"] = Heuristic.INPUT

        if self.env["verbosity"] >= 3:
            np.set_printoptions(threshold=sys.maxsize)
            np.set_printoptions(linewidth=np.inf)

        if env is not None:
            self.env = {**self.env, **env}
        self.ivarmap = {}

        self.tables = []
        super().__init__(cpm_model=cpm_model, subsolver=subsolver, lazy=True)

    def log(self, *mess, verbosity=1, end="\n"):
        if verbosity <= self.env["verbosity"]:
            print(*mess, end=end)

    def show_env(self):
        self.log(", ".join(f"{k}={self.env[k]}" for k in ["shrink", "heuristic"]), verbosity=0)
        # log(f"cuts = {env['cuts']}", env=env, verbosity=1)
        self.log(f"n_cuts = {sum(len(c['cut']) > 0 for c in self.env['cuts'])}", verbosity=0)
        self.log(f"n_unexplained = {list(len(c['cut']) for c in self.env['cuts']).count(0)}", verbosity=0)
        self.log(
            "cut cardinalities/strengths:",
            ", ".join(f"{len(c['cut'])}" for c in self.env["cuts"]),
            verbosity=2,
        )

    def choose(self, A, T_enc, R, heuristic=Heuristic.GREEDY):
        self.log(f"Choose from {sorted(A)} from remaining choices {R}", verbosity=2)
        if len(A) == 0:
            return None
        match heuristic:
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
                return choice if choice is not None else self.choose(A, T_enc, R, heuristic=Heuristic.GREEDY)

    def shrink(self, C, T):
        for i in C:
            self.log(f"shrinking {i} in {C}", verbosity=3)

            if len(C) <= 1:  # slightly different from
                return C

            S = set.intersection(*[rows(T, l) for l in (C - {i})])
            if len(S) == 0:
                C = C - {i}
        return C

    def explain(self, A_enc, T_enc):
        """The `explain_frac` alg."""

        # TODO convert T_enc to set of tuples?
        self.log(f"Explain")
        self.log("", np.array(A_enc), verbosity=2)
        self.log(np.array(T_enc), verbosity=2)

        X = set()  # columns added to cut
        R = set(range(len(T_enc)))  # remaining columns
        W = set(i for i, a in enumerate(A_enc) if a == 1.0)
        F = set(i for i, a in enumerate(A_enc) if 0.0 < a < 1.0)

        self.log(f"W = {sorted(W)}", verbosity=3)
        self.log(f"F = {sorted(F)}", verbosity=3)

        for iteration in itertools.count(start=1):
            if not len(R):
                break
            if W <= X:
                self.log("  unexplainable")
                self.log(f"    because {W} <= {X}", verbosity=3)
                # if is_integer_solution(A_enc):
                # assert not is_integer_solution(A_enc), f"Could not explain an integer solution: {A_enc}"
                return set()  # no cut

            # choose some col which is 1
            self.log("A", A_enc, X, verbosity=3)
            i = self.choose(W - X, T_enc, R, heuristic=self.env["heuristic"])
            self.log(f"chosen {i}", verbosity=3)

            R = R.intersection(rows(T_enc, i))
            X.add(i)

            # Loop termination for debug purposes
            assert iteration <= self.env.get("max_iterations", iteration)

        self.log(f"  by explanation of size ({len(X)}): {X}")
        if self.env.get("shrink", True):
            size = len(X)
            X = self.shrink(X, T_enc)
            if len(X) < size:
                self.log(f"  shrunk to size ({len(X)}): {X}")

        self.log(f"  cut == {X}")
        return X

    def log_explanation(self, explanation, frm):
        if "cuts" in self.env:
            if self.env["debug"]:
                assert explanation not in (c["cut"] for c in self.env["cuts"]), (
                    f"Already found cut {explanation} previously in {self.env['cuts']}"
                )
            self.env["cuts"].append({"cut": explanation, "shrunk": 0, "from": frm})

    def get_solution_callback(self):
        all_xs = get_x_enc(self.ivarmap)

        def solution_callback(what, where):
            try:
                x_enc_a = None
                frm = None
                match where:
                    case GRB.Callback.MIPNODE:
                        # Optimal solution to LP relaxation
                        if what.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
                            x_enc_a = {
                                x_enc_i: what.cbGetNodeRel(self.solver_var(x_enc_i)) for x_enc_i in all_xs
                            }
                            frm = "MIPNODE-OPT"
                        else:
                            return
                    case GRB.Callback.MIPSOL:  # Integer solution
                        x_enc_a = {
                            x_enc_i: what.cbGetSolution(self.solver_var(x_enc_i)) for x_enc_i in all_xs
                        }
                        frm = "MIPSOL"
                        assert is_integer_solution(x_enc_a.values()), (
                            f"Expected integer solution for MIP, but got {x_enc_a}"
                        )
                        # Recommended way to convert fractional integer solution into Boolean, then using `int` to convert to 01
                        x_enc_a = {x_enc_i: int(a_enc_i > 0.5) for x_enc_i, a_enc_i in x_enc_a.items()}
                    case _:
                        return

                self.log(frm, x_enc_a, verbosity=2)

                # If fully integer, we can check if the tables are feasible yet
                for X_enc, T_enc in self.tables:
                    A_enc = [x_enc_a[x_enc_i] for x_enc_i in X_enc]

                    # TODO figure out when can be skipped
                    # if frm == "MIPNODE-OPT" and is_integer_solution(A_enc) and
                    if A_enc in T_enc.tolist():
                        continue

                    # encode assignment
                    explanation = self.explain(A_enc, T_enc)
                    self.log_explanation(explanation, frm)

                    if explanation:
                        grbs = [self.solver_var(X_enc[c]) for c in explanation]
                        what.cbLazy(gp.quicksum(grbs) <= len(grbs) - 1)
                    elif frm == "MIPSOL":  # unsat
                        what.cbLazy(False)

            except Exception as e:
                what._callback_exception = e
                what.terminate()

        return solution_callback

    def solve(self, time_limit=None, solution_callback=None, **kwargs):
        """
        Call the gurobi solver with cut generation
        """

        self.log("Solving.. ")
        self.native_model.Params.LazyConstraints = 1
        if self.env["verbosity"] >= 4:
            self.native_model.Params.LogFile = "/tmp/gurobi.log"
            self.native_model.Params.OutputFlag = 1
            self.native_model.write("/tmp/gurobi.lp")

        hassol = super().solve(solution_callback=self.get_solution_callback())
        # TODO recheck https://or.stackexchange.com/questions/12591/ensure-gurobi-uses-callback-on-all-feasible-solutions It looks like if the solution at the end of the root node is integer, gurobi doesn't pass through callbacks for fractional solutions.

        return hassol

    def transform(self, cpm_expressions):
        cpm_cons = []  # all but tables
        for cpm_expr in cpm_expressions:
            if hasattr(cpm_expr, "name") and cpm_expr.name == "table":
                cpm_expr_tfs = [cpm_expr]
            else:
                cpm_expr_tfs = super().transform(cpm_expr)

            for cpm_expr_tf in cpm_expr_tfs:
                if cpm_expr_tf.name == "table":
                    X, T = cpm_expr_tf.args
                    self.log("X =", ", ".join(f"{x} in {x.lb}..{x.ub}" for x in X), verbosity=0)
                    self.log("T =", verbosity=1)
                    self.log(T, verbosity=1)
                    T_enc = encode(X, T)
                    self.log("T_enc =", verbosity=1)
                    self.log(T_enc, verbosity=1)

                    for x in X:
                        x_enc, exactly_one_con = cp.transformations.int2bool._encode_int_var(
                            self.ivarmap, x, "direct"
                        )
                        expr, k = x_enc.encode_term()
                        # TODO if only BV, then need to assign (but no need to assign if decoding constraint present)
                        cpm_cons += self.transform(
                            [*exactly_one_con, cp.sum(c * b for c, b in expr) + k == x]
                        )
                    X_enc = get_x_enc(self.ivarmap)
                    self.tables.append((X_enc, T_enc))
                else:
                    cpm_cons.append(cpm_expr_tf)
        return cpm_cons


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
    assert pytest.main() == pytest.ExitCode.OK
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
