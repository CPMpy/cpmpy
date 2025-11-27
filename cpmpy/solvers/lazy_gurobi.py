import itertools
import math
import pprint
import sys
import time
from enum import Enum

import gurobipy as gp
import numpy as np
from gurobipy import GRB

import cpmpy as cp
from cpmpy.expressions.variables import NegBoolView
from cpmpy.solvers.gurobi import CPM_gurobi

INDEX = 1


def show_set(S, index=INDEX):
    return f"{{{', '.join(str(s + index) for s in sorted(S))}}}"


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
    try:
        X[a - lb] = 1
    except IndexError:
        # Fortress1-03_c25 case
        # ['0..0', '0..1']
        # [[0, 1], [1, 0]]
        pass
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
        self.env["shrink"] = False
        self.env["explain_fractional"] = True
        self.env["cuts"] = []
        self.env["max_iterations"] = None
        self.indent = 0

        if env is not None:
            self.env = {**self.env, **env}

        if self.env["verbosity"] >= 3:
            np.set_printoptions(threshold=sys.maxsize)
            np.set_printoptions(linewidth=np.inf)
            pprint.pprint(env)

        self.ivarmap = {}

        self.tables = []
        super().__init__(lazy=True, **kwargs)

    def log(self, *mess, verbosity=1, end="\n", indent=None):
        if verbosity <= self.env["verbosity"]:
            indent = self.indent if indent is None else indent
            print(" " * indent * 2, *mess, end=end, flush=self.env["debug"])

    def stats(self):
        self.log(
            ", ".join(f"{k}={self.env[k]}" for k in ["shrink", "heuristic", "explain_fractional"]),
            verbosity=0,
        )
        import pandas as pd

        cuts_df = pd.DataFrame.from_dict(self.env["cuts"])
        self.log(cuts_df.drop(["failure"], axis=1, errors="ignore"), verbosity=2)
        # self.log(pprint.pformat(self.env["cuts"]), verbosity=2)
        cuts = [c for c in self.env["cuts"] if "size" in c]
        cuts_mipsol = [c for c in cuts if c["from"] == "MIPSOL"]
        assert all(c["size"] > 0 for c in cuts_mipsol)
        cuts_mipnode = [c for c in cuts if c["from"] == "MIPNODE-OPT"]
        cuts_mipnode_exp = [c for c in cuts_mipnode if c["size"] > 0]
        cuts_mipnode_unexp = [c for c in cuts_mipnode if c["size"] == 0]
        self.log(f"cb_time = {self.env['cb_time']}")
        self.log(f"cuts (MIPSOL) = {len(cuts_mipsol)}")
        self.log(f"cuts (MIPNODE, explained) = {len(cuts_mipnode_exp)}")
        self.log(f"cuts (MIPNODE, unexplainable) = {len(cuts_mipnode_unexp)}")
        return {
            "cb_time": self.env["cb_time"],
            "n_cuts": len(cuts_mipsol),
            "n_cuts_explained": len(cuts_mipnode_exp),
            "n_cuts_unexplained": len(cuts_mipnode_unexp),
        }

    def choose(self, A, T_enc, R, heuristic=Heuristic.GREEDY):
        self.log(f"Choose from {show_set(A)} from remaining choices {show_set(R)}", verbosity=2)
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
            self.log(f"shrinking {i + INDEX} in {show_set(C)}", verbosity=3)

            if len(C) <= 1:  # slightly different from
                return C

            S = set.intersection(*[rows(T, l) for l in (C - {i})])
            if len(S) == 0:
                C = C - {i}
        return C

    def explain(self, A_enc, T_enc, parts, frm=None):
        """The `explain_frac` alg."""

        parts_ = [0]
        for p in parts:
            parts_.append(parts_[-1] + p)
        parts = parts_[1:]

        def p(i):
            return next(l for l in range(len(parts)) if i < parts[l])

        # TODO convert T_enc to set of tuples?
        self.log("Explain", end="\n")

        self.log("", np.array(A_enc), verbosity=1)
        self.log(np.array(T_enc), verbosity=2)
        self.log("")
        assert len(A_enc) == len(T_enc.T)

        self.env["cuts"].append({"from": frm})
        if self.env["debug"]:
            self.env["cuts"][-1]["failure"] = A_enc

        m = len(T_enc)  # number of cols
        W = set(i for i, a in enumerate(A_enc) if a == 1.0)  #
        F = set(i for i, a in enumerate(A_enc) if 0.0 < a < 1.0)
        X = set()  # columns added to cut
        R = set(range(len(T_enc)))  # remaining columns
        D = set(r for r in range(m) if cols(T_enc, r) <= W.union(F))  # difficult rows; either frac/whole
        U = set(i for i in F if all(T_enc[r, i] == 0 for r in D))  # frac except difficult


        self.log(f"W = {show_set(W)}", verbosity=3)
        self.log(f"F = {show_set(F)}", verbosity=3)
        self.log(f"D = {show_set(D)}", verbosity=3)
        self.log(f"U = {show_set(U)}", verbosity=3)

        if F:
            if not U:
                self.log("  unexplainable")
                self.log("    because U is empty", verbosity=3)
                return
            else:
                i = self.choose(U, T_enc, R, heuristic=self.env["heuristic"])
                self.log(f"chosen {i + INDEX}", verbosity=3, indent=self.indent + 2)

                # TODO [peter] should be T_hat[i]?
                R = rows(T_enc, i)
                X = {i}
                V = {p(i)}
                s = 0
        else:
            R = set(range(m))
            X = set()
            V = set()
            s = -1

        for iteration in itertools.count(start=1):
            self.check_max_iterations(iteration)
            self.indent = 1
            self.log(f"X = {show_set(X)}", verbosity=3)
            self.log(f"V = {show_set(V)}", verbosity=3)
            self.log(f"R = {show_set(R)}", verbosity=3)
            self.log(f"s = {s}", verbosity=3)
            if not R:
                break
            l = next(l for l in set(range(len(parts))) - V)
            self.log(f"Choose integer l = {INDEX + l}", verbosity=3)
            V.add(l)
            s += 1

            def C(v, l):
                self.log(f"C({v}, {l})", verbosity=3)
                return set(i for i in range(len(v)) if v[i] > 0 and p(i) == l)

            sets = [rows(T_enc, i) for i in C(A_enc, l)]
            R.intersection_update(set.union(*sets) if sets else set())
            X = X.union(C(A_enc, l))

        self.indent = 1
        self.log(f"chosen {show_set(V)}", verbosity=3)

        self.indent = 0
        self.log(f"  by explanation of size ({len(X)}): {show_set(X)}")

        if self.env["debug"]:
            self.env["cuts"][-1]["cut"] = X.copy()

        if self.env["shrink"]:
            X_shrunk = self.shrink(X, T_enc)
            shrunk = len(X) - len(X_shrunk)
            if shrunk:
                self.log(f"  shrunk by {shrunk}: {show_set(X)} --> {show_set(X_shrunk)}")
                if self.env["debug"]:
                    self.env["cuts"][-1] = {
                        **self.env["cuts"][-1],
                        "pre_shrunk": self.env["cuts"][-1]["cut"],
                        "cut": X_shrunk,
                    }
                self.env["cuts"][-1]["shrunk"] = shrunk

        self.env["cuts"][-1]["size"] = len(X)
        self.log(f"  cut == {show_set(X)}")
        return X

    def check_max_iterations(self, i):
        # Loop termination for debug purposes
        if self.env["max_iterations"] is not None:
            assert i <= self.env["max_iterations"]

    def get_solution_callback(self):
        all_xs = {x_enc_i for x_enc, _, _ in self.tables for x_enc_i in x_enc}

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
                        if self.env["debug"]:
                            assert is_integer_solution(x_enc_a.values()), (
                                f"Expected integer solution for MIP, but got {x_enc_a}"
                            )
                    case _:
                        return

                if frm == "MIPSOL":
                    # Recommended way to convert fractional integer solution into Boolean, then using `int` to convert to 01
                    x_enc_a = {x_enc_i: int(a_enc_i > 0.5) for x_enc_i, a_enc_i in x_enc_a.items()}

                self.log(frm, x_enc_a, verbosity=2)

                # If fully integer, we can check if the tables are feasible yet
                for X_enc, T_enc, parts in self.tables:
                    A_enc = [x_enc_a[x_enc_i] for x_enc_i in X_enc]

                    # TODO figure out when can be skipped
                    # if frm == "MIPNODE-OPT" and is_integer_solution(A_enc) and
                    if A_enc in T_enc.tolist():
                        continue

                    # encode assignment
                    explanation = self.explain(A_enc, T_enc, parts, frm=frm)

                    if explanation:
                        self.log(
                            f"  cons == {' + '.join(X_enc[c].name for c in explanation)} < {len(explanation)}",
                        )
                        grbs = [self.solver_var(X_enc[c]) for c in explanation]
                        what.cbLazy(gp.quicksum(grbs) <= len(grbs) - 1.0)
                    elif frm == "MIPSOL":  # unsat
                        self.log("INFEASIBLE")
                        raise Infeasible

                    cb_time = time.time() - cb_time
                    self.log(f"end callback, time = {cb_time}")
                    self.env["cb_time"] += cb_time
                    self.check_max_iterations(len(self.env["cuts"]))
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

        if getattr(self.native_model, "_callback_exception", None):
            raise self.grb_model._callback_exception or Exception(
                "Gurobi was interrupted (perhaps the solution callback called model.terminate())"
            )

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

                x_encs = [self.ivarmap[x.name]._xs for x in X]
                parts = [len(x_enc) for x_enc in x_encs]
                X_enc = [x_enc_i for x_enc in x_encs for x_enc_i in x_enc]
                self.tables.append((X_enc, T_enc, parts))
            else:
                cpm_cons.append(cpm_expr)
        return cpm_cons

    def _check_repeat_failure(self):
        if self.env["debug"] and False:
            prev = next(
                (c for c in self.env["cuts"][:-1] if self.env["cuts"][-1]["failure"] == c["failure"]),
                None,
            )

            assert prev is None, f"""Encountered same failure twice:

        {pprint.pformat(self.env["cuts"][-1])}

        should have been prevented by earlier cut:

        {pprint.pformat(prev)}
        """
