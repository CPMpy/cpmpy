import sys
import time
import math
import gurobipy as gp
from gurobipy import GRB
import itertools
import numpy as np
import cpmpy as cp
from cpmpy.expressions.variables import NegBoolView

from cpmpy.solvers.gurobi import CPM_gurobi
from enum import Enum


class Infeasible(Exception):
    pass


def rows(T, i, j=1):
    """Row indices where T[i]==j"""
    return set(int(i) for i in np.where(T[:, i] == j)[0].flatten())


def cols(T, i, j=1):
    """Col indices where T[i]==j"""
    return set(int(i) for i in np.where(T[i, :] == j)[0].flatten())


def is_integer_solution(A_enc):
    return all(math.isclose(a_enc_i, a_enc_i > 0.5, abs_tol=1e-5) for a_enc_i in A_enc)


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


class Heuristic(Enum):
    INPUT = 1
    GREEDY = 2
    REDUCE = 3


class CPM_lazy_gurobi(CPM_gurobi):
    def __init__(self, env=None, **kwargs):
        self.env = {}
        self.env["debug"] = False
        self.env["debug_unlucky"] = False  # TODO re-enable
        self.env["verbosity"] = 0
        self.env["heuristic"] = Heuristic.INPUT
        self.env["shrink"] = True
        self.env["explain_fractional"] = True
        self.env["max_iterations"] = None

        if self.env["verbosity"] >= 3:
            np.set_printoptions(threshold=sys.maxsize)
            np.set_printoptions(linewidth=np.inf)

        if env is not None:
            self.env = {**self.env, **env}
        self.ivarmap = {}

        self.tables = []
        super().__init__(lazy=True, **kwargs)

    def log(self, *mess, verbosity=1, end="\n"):
        if verbosity <= self.env["verbosity"]:
            print(*mess, end=end)

    def stats(self):
        self.log(
            ", ".join(f"{k}={self.env[k]}" for k in ["shrink", "heuristic", "explain_fractional"]),
            verbosity=0,
        )
        # log(f"cuts = {env['cuts']}", env=env, verbosity=1)
        cuts = [c for c in self.env["cuts"] if len(c["cut"]) > 0]
        cuts_mipsol = [c for c in self.env["cuts"] if c["from"] == "MIPSOL"]
        assert all(len(c["cut"]) > 0 for c in cuts_mipsol)
        cuts_mipnode = [c for c in self.env["cuts"] if c["from"] == "MIPNODE-OPT"]
        cuts_mipnode_exp = [c for c in cuts_mipnode if len(c["cut"]) > 0]
        cuts_mipnode_unexp = [c for c in cuts_mipnode if len(c["cut"]) == 0]
        self.log(f"cb_time = {self.env['cb_time']}")
        self.log(f"cuts (MIPSOL) = {len(cuts_mipsol)}")
        self.log(f"cuts (MIPNODE, explained) = {len(cuts_mipnode_exp)}")
        self.log(f"cuts (MIPNODE, unexplainable) = {len(cuts_mipnode_unexp)}")
        self.log(
            "cut cardinalities/strengths:",
            ", ".join(f"{len(c['cut'])}" for c in self.env["cuts"]),
            verbosity=2,
        )
        return {
            "cb_time": self.env["cb_time"],
            "n_cuts": len(cuts_mipsol),
            "n_cuts_explained": len(cuts_mipnode_exp),
            "n_cuts_unexplained": len(cuts_mipnode_unexp),
        }

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
        self.log(f"Explain", end="\n")
        # self.log("", A_enc, verbosity=1)
        self.log("", sorted(cols(np.array([A_enc]), 0)))
        self.log(np.array(T_enc), verbosity=2)
        self.log("")

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
            if self.env["max_iterations"] is not None:
                assert iteration <= self.env["max_iterations"]

        self.log(f"  by explanation of size ({len(X)}): {sorted(X)}")
        if self.env["shrink"]:
            size = len(X)
            X = self.shrink(X, T_enc)
            if len(X) < size:
                self.log(f"  shrunk to size ({len(X)}): {sorted(X)}")

        self.log(f"  cut == {sorted(X)}")
        return X

    def log_explanation(self, explanation, frm):
        if "cuts" in self.env:
            if self.env["debug"]:
                assert explanation not in (c["cut"] for c in self.env["cuts"]), (
                    f"Already found cut {explanation} previously in {self.env['cuts']}"
                )
            self.env["cuts"].append({"cut": explanation, "shrunk": 0, "from": frm})

    def get_solution_callback(self):
        all_xs = self.get_x_encs(self.ivarmap.keys())

        def solution_callback(what, where):
            cb_time = time.time()

            try:
                x_enc_a = None
                frm = None

                def cbGetVal(cpm_var, cbGet):
                    if isinstance(cpm_var, NegBoolView):
                        return 1.0 - cbGet(self.solver_var(~cpm_var))
                    return cbGet(self.solver_var(cpm_var))

                match where:
                    case GRB.Callback.MIPNODE:
                        if not self.env["explain_fractional"]:
                            return
                        # Optimal solution to LP relaxation
                        if what.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
                            x_enc_a = {x_enc_i: cbGetVal(x_enc_i, what.cbGetNodeRel) for x_enc_i in all_xs}
                            frm = "MIPNODE-OPT"
                        else:
                            return
                    case GRB.Callback.MIPSOL:  # Integer solution
                        x_enc_a = {x_enc_i: cbGetVal(x_enc_i, what.cbGetSolution) for x_enc_i in all_xs}
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
                        self.log(
                            f"  cons == {' + '.join(X_enc[c].name for c in explanation)} < {len(explanation)}",
                        )
                        grbs = [self.solver_var(X_enc[c]) for c in explanation]
                        what.cbLazy(gp.quicksum(grbs) <= len(grbs) - 1)
                    elif frm == "MIPSOL":  # unsat
                        self.log("INFEASIBLE")
                        raise Infeasible
                        # what.cbLazy(1 <= 0)

                if self.env["max_iterations"] is not None:
                    assert len(self.env["cuts"]) <= self.env["max_iterations"]

                self.env["cb_time"] += time.time() - cb_time
                self.log("end callback")
            except Exception as e:
                what._callback_exception = e
                what.terminate()

        return solution_callback

    def solve(self, time_limit=None, solution_callback=None, **kwargs):
        """
        Call the gurobi solver with cut generation
        """

        self.env["cuts"] = []
        self.env["cb_time"] = 0.0

        assert solution_callback is None, "For now, no solution_callback in `CPM_lazy_gurobi`"

        self.log("Solving.. ")
        self.native_model.Params.LazyConstraints = 1
        if self.env["verbosity"] >= 4:
            self.native_model.Params.LogFile = "/tmp/gurobi.log"
            self.native_model.Params.OutputFlag = 1
            self.native_model.write("/tmp/gurobi.lp")

        try:
            hassol = super().solve(
                solution_callback=self.get_solution_callback(), time_limit=time_limit, **kwargs
            )
        except Infeasible:
            hassol = False

        for field, stat in self.stats().items():
            print(f"c Stat={field}={stat}")

        # TODO recheck https://or.stackexchange.com/questions/12591/ensure-gurobi-uses-callback-on-all-feasible-solutions It looks like if the solution at the end of the root node is integer, gurobi doesn't pass through callbacks for fractional solutions.

        return hassol

    def get_x_encs(self, X):
        return [x_enc_i for x in X for x_enc_i in self.ivarmap[x]._xs]

    def transform(self, cpm_expressions):
        cpm_cons = []  # all but tables
        cpm_expressions = super().transform(cpm_expressions)
        for cpm_expr in cpm_expressions:
            if cpm_expr.name == "table":
                X, T = cpm_expr.args
                self.log("X =", ", ".join(f"{x} in {x.lb}..{x.ub}" for x in X), verbosity=2)
                self.log("T =", verbosity=2)
                self.log(T, verbosity=2)
                T_enc = encode(X, T)
                self.log("T_enc =", verbosity=2)
                self.log(T_enc, verbosity=2)

                for x in X:
                    x_enc, exactly_one_con = cp.transformations.int2bool._encode_int_var(
                        self.ivarmap, x, "direct", csemap=self._csemap
                    )
                    expr, k = x_enc.encode_term()
                    # TODO if only BV, then need to assign (but no need to assign if decoding constraint present)
                    cpm_cons += self.transform([*exactly_one_con, cp.sum(c * b for c, b in expr) + k == x])
                X_enc = self.get_x_encs(x.name for x in X)
                self.tables.append((X_enc, T_enc))
            else:
                cpm_cons.append(cpm_expr)
        return cpm_cons
