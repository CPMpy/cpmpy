import time
from typing import Optional

import cpmpy as cp
from cpmpy.tools.benchmark import _bytes_as_mb, _bytes_as_gb, _mib_as_bytes

from ..runner.runner import Runner
from .base import Observer


# largest coefficient magnitude considered safe for HiGHS default matrix scaling;
# the observed false-UNSAT instances had coefficients ~1e9 (drops occur when scaled
# entries fall below small_matrix_value=1e-9), 1e6 leaves 3 orders of magnitude margin
_HIGHS_SAFE_COEFFICIENT_MAGNITUDE = 1e6


def _max_abs_constant(model: cp.Model, stop_at=None):
    """Largest absolute numeric constant in the model (weights, bounds, ...).
       With `stop_at`, returns early once that magnitude is reached."""
    import numpy as np
    from cpmpy.expressions.core import Expression
    from cpmpy.expressions.variables import _NumVarImpl

    mx = 0
    stack = list(model.constraints)
    if model.has_objective():
        stack.append(model.objective_)
    while stack:
        e = stack.pop()
        if isinstance(e, _NumVarImpl):
            continue
        if isinstance(e, Expression):
            stack.extend(e.args)
        elif isinstance(e, (list, tuple, np.ndarray)):
            stack.extend(e)
        elif isinstance(e, (int, float, np.integer, np.floating)):
            mx = max(mx, abs(e))
            if stop_at is not None and mx >= stop_at:
                return mx
    return mx


class SolverArgsObserver(Observer):

    def __init__(self, **kwargs):
        self.time_limit = None
        self.mem_limit = None
        self.seed = None
        self.intermediate = False
        self.cores = 1
        self.mem_limit = None
        self.kwargs = dict()

    def observe_init(self, runner: Runner):
        self.time_limit = runner.time_limit
        self.mem_limit = runner.mem_limit
        self.seed = runner.seed
        self.intermediate = runner.intermediate
        self.cores = runner.cores
        self.mem_limit = runner.mem_limit
        self.kwargs = runner.kwargs

    def _ortools_arguments(
            self,
            runner: Runner,
            model: cp.Model,
            cores: Optional[int] = None,
            seed: Optional[int] = None,
            intermediate: bool = False,
            **kwargs
        ):
        # https://github.com/google/or-tools/blob/stable/ortools/sat/sat_parameters.proto
        res = dict()

        if cores == 1:
            # https://github.com/google/or-tools/blob/1c5daab55dd84bca7149236e4b4fa009e5fd95ca/ortools/flatzinc/cp_model_fz_solver.cc#L1688
            res |= {
                "interleave_search": True,
                "use_rins_lns": False,
            }
            if not model.has_objective():
                res |= {"num_violation_ls": 1}

        if cores is not None:
            if cores > 1 and cores < 8:
                # Bump up to 8 workers
                res |= {"num_search_workers": 8}
            else:
                res |= {"num_search_workers": cores}
        if seed is not None:
            res |= {"random_seed": seed}

        if intermediate and model.has_objective():
            # Define custom ORT solution callback, then register it
            from ortools.sat.python import cp_model as ort

            class OrtSolutionCallback(ort.CpSolverSolutionCallback):
                """
                    For intermediate objective printing.
                """

                def __init__(self):
                    super().__init__()
                    self.__start_time = time.time()
                    self.__solution_count = 1

                def on_solution_callback(self):
                    """Called on each new solution."""

                    current_time = time.time()
                    obj = int(self.ObjectiveValue())
                    runner.print_comment("Solution %i, time = %0.4fs" %
                                 (self.__solution_count, current_time - self.__start_time))
                    runner.observe_intermediate(obj)
                    self.__solution_count += 1

                def solution_count(self):
                    """Returns the number of solutions found."""
                    return self.__solution_count

            # Register the callback
            res |= {"solution_callback": OrtSolutionCallback()}

        def internal_options(solver: "CPM_ortools"):
            if cores == 1:
                # https://github.com/google/or-tools/blob/1c5daab55dd84bca7149236e4b4fa009e5fd95ca/ortools/flatzinc/cp_model_fz_solver.cc#L1688
                solver.ort_solver.parameters.subsolvers.extend(["default_lp", "max_lp", "quick_restart"])
                if not model.has_objective():
                    solver.ort_solver.parameters.subsolvers.append("core_or_no_lp")
                if len(solver.ort_model.proto.search_strategy) != 0:
                    solver.ort_solver.parameters.subsolvers.append("fixed")

        return res, internal_options

    def _exact_arguments(
            self,
            seed: Optional[int] = None,
            **kwargs
        ):
        # Documentation: https://gitlab.com/JoD/exact/-/blob/main/src/Options.hpp?ref_type=heads
        res = dict()
        if seed is not None:
            res |= {"seed": seed}

        return res, None

    def _choco_arguments(self):
        # Documentation: https://github.com/chocoteam/pychoco/blob/master/pychoco/solver.py
        return {}, None

    def _z3_arguments(
            self,
            model: cp.Model,
            cores: int = 1,
            seed: Optional[int] = None,
            mem_limit: Optional[int] = None,
            **kwargs
        ):
        # Documentation: https://microsoft.github.io/z3guide/programming/Parameters/
        # -> is outdated, just let it crash and z3 will report the available options

        res = dict()

        if model.has_objective():
            # Opt does not seem to support setting random seed or max memory
            pass
        else:
            # Sat parameters
            if cores is not None:
                res |= {"threads": cores}  # TODO what with hyperthreadding, when more threads than cores
            if seed is not None:
                res |= {"random_seed": seed}
            if mem_limit is not None:
                res |= {"max_memory": _bytes_as_mb(mem_limit)}

        return res, None

    def _minizinc_arguments(
            self,
            solver: str,
            cores: Optional[int] = None,
            seed: Optional[int] = None,
            **kwargs
        ):
        # Documentation: https://minizinc-python.readthedocs.io/en/latest/api.html#minizinc.instance.Instance.solve
        res = dict()
        if cores is not None:
            res |= {"processes": cores}
        if seed is not None:
            res |= {"random_seed": seed}

        # if solver.endswith("gecode"):
        # Documentation: https://www.minizinc.org/doc-2.4.3/en/lib-gecode.html
        # elif solver.endswith("chuffed"):
        # Documentation:
        # - https://www.minizinc.org/doc-2.5.5/en/lib-chuffed.html
        # - https://github.com/chuffed/chuffed/blob/develop/chuffed/core/options.h

        return res, None

    def _gurobi_arguments(
            self,
            runner: Runner,
            model: cp.Model,
            cores: Optional[int] = None,
            seed: Optional[int] = None,
            mem_limit: Optional[int] = None,
            intermediate: bool = False,
            solver_params: Optional[dict] = None,
            **kwargs
        ):
        # Documentation: https://www.gurobi.com/documentation/9.5/refman/parameters.html#sec:Parameters
        res = dict()
        if cores is not None:
            res |= {"Threads": cores}
        if seed is not None:
            res |= {"Seed": seed}
        if mem_limit is not None:
            res |= {"MemLimit": _bytes_as_gb(mem_limit)}
        if solver_params:
            res |= solver_params

        if intermediate and model.has_objective():

            class GurobiSolutionCallback:
                def __init__(self, model: cp.Model, runner: Runner):
                    self.__start_time = time.time()
                    self.__solution_count = 0
                    self.model = model
                    self.runner = runner

                def callback(self, *args, **kwargs):
                    current_time = time.time()
                    model, state = args

                    # Callback codes: https://www.gurobi.com/documentation/current/refman/cb_codes.html#sec:CallbackCodes

                    from gurobipy import GRB
                    # if state == GRB.Callback.MESSAGE: # verbose logging
                    #     print_comment("log message: " + str(model.cbGet(GRB.Callback.MSG_STRING)))
                    if state == GRB.Callback.MIP:  # callback from the MIP solver
                        if model.cbGet(GRB.Callback.MIP_SOLCNT) > self.__solution_count:  # do we have a new solution?

                            obj = int(model.cbGet(GRB.Callback.MIP_OBJBST))
                            self.runner.print_comment("Solution %i, time = %0.4fs" %
                                                (self.__solution_count, current_time - self.__start_time))
                            self.runner.observe_intermediate(obj)
                            self.__solution_count = model.cbGet(GRB.Callback.MIP_SOLCNT)

            res |= {"solution_callback": GurobiSolutionCallback(model, runner).callback}

        return res, None

    def _cpo_arguments(
            self,
            runner: Runner,
            model: cp.Model,
            cores: Optional[int] = None,
            seed: Optional[int] = None,
            intermediate: bool = False,
            **kwargs
        ):
        # Documentation: https://ibmdecisionoptimization.github.io/docplex-doc/cp/docplex.cp.parameters.py.html#docplex.cp.parameters.CpoParameters
        res = dict()
        if cores is not None:
            res |= {"Workers": cores}
        if seed is not None:
            res |= {"RandomSeed": seed}

        if intermediate and model.has_objective():
            from docplex.cp.solver.solver_listener import CpoSolverListener
            class CpoSolutionCallback(CpoSolverListener):

                def __init__(self):
                    super().__init__()
                    self.__start_time = time.time()
                    self.__solution_count = 1
                    self.runner = runner

                def result_found(self, solver, sres):
                    current_time = time.time()
                    obj = sres.get_objective_value()
                    if obj is not None:
                        self.runner.print_comment("Solution %i, time = %0.4fs" %
                                           (self.__solution_count, current_time - self.__start_time))
                        self.runner.observe_intermediate(int(obj))
                        self.__solution_count += 1

                def solution_count(self):
                    """Returns the number of solutions found."""
                    return self.__solution_count

            # Register the callback
            # docplex expects a listener class, not an instance.
            res |= {"solution_callback": CpoSolutionCallback}

        return res, None

    def _cplex_arguments(
        self,
        cores: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs
    ):
        res = dict()
        if cores is not None:
            res |= {"threads": cores}
        if seed is not None:
            res |= {"randomseed": seed}

        return res, None

    def _hexaly_arguments(
        self,
        runner: Runner,
        model: cp.Model,
        cores: Optional[int] = None,
        seed: Optional[int] = None,
        intermediate: bool = False,
        **kwargs
    ):
        res = dict()
        # res |= {"nb_threads": cores}
        # res |= {"seed": seed}

        if intermediate and model.has_objective():
            # Define custom Hexaly solution callback, then register it

            class HexSolutionCallback:

                def __init__(self, runner: Runner):
                    self.__start_time = time.time()
                    self.__solution_count = 0
                    self.runner = runner

                def on_solution_callback(self, optimizer, cb_type):
                    """Called on each new solution."""
                    # check if solution with different objective (or if verbose)
                    current_time = time.time()
                    obj = optimizer.model.objectives[0].value
                    self.runner.print_comment("Solution %i, time = %0.4fs" %
                                       (self.__solution_count, current_time - self.__start_time))
                    self.runner.observe_intermediate(int(obj))
                    self.__solution_count += 1

                def solution_count(self):
                    return self.__solution_count

            # Register the callback
            res |= {"solution_callback": HexSolutionCallback(runner).on_solution_callback}

        return res, None

    def _highs_arguments(
        self,
        runner: Runner,
        model: cp.Model,
        cores: Optional[int] = None,
        seed: Optional[int] = None,
        intermediate: bool = False,
        **kwargs
    ):
        res = dict()
        # HiGHS is a float MIP solver; defaults trade exactness for speed, which is
        # unsound for competition answers:
        # - default mip_rel_gap=1e-4 lets HiGHS report near-optimal solutions as OPTIMAL.
        #   CPMpy objectives are integer-valued, so an absolute gap < 1 already proves
        #   optimality: rel_gap=0 + abs_gap=0.99 is exact yet stops earlier than the
        #   default gaps (no need to close the gap to 0).
        res |= {
            "mip_rel_gap": 0.0,
            "mip_abs_gap": 0.99,
        }
        # - default small_matrix_value=1e-9 drops scaled matrix entries on instances with
        #   large coefficients, turning satisfiable instances into false UNSAT. Lowering
        #   it perturbs the search on all instances, so only do it when coefficients are
        #   large enough for scaling to produce sub-1e-9 entries.
        if _max_abs_constant(model, stop_at=_HIGHS_SAFE_COEFFICIENT_MAGNITUDE) \
                >= _HIGHS_SAFE_COEFFICIENT_MAGNITUDE:
            res |= {"small_matrix_value": 1e-12}  # minimum allowed by HiGHS
        if cores is not None:
            res |= {"threads": cores}
        if seed is not None:
            res |= {"random_seed": seed}

        internal_options = None
        if intermediate and model.has_objective():

            class HighsSolutionCallback:
                def __init__(self, runner: Runner, mip_solution_callback_type):
                    self.__start_time = time.time()
                    self.__solution_count = 0
                    self.runner = runner
                    self.mip_solution_callback_type = mip_solution_callback_type

                def callback(self, callback_type, message, data_out, data_in, user_data):
                    if callback_type != self.mip_solution_callback_type:
                        return

                    current_time = time.time()
                    obj = data_out.objective_function_value
                    self.runner.print_comment("Solution %i, time = %0.4fs" %
                                       (self.__solution_count, current_time - self.__start_time))
                    self.runner.observe_intermediate(int(obj))
                    self.__solution_count += 1

            def internal_options(solver):
                from highspy import cb as hscb

                callback_type = hscb.HighsCallbackType.kCallbackMipSolution
                callback = HighsSolutionCallback(runner, callback_type)
                solver._cpbenchy_highs_solution_callback = callback
                solver.highs.setCallback(callback.callback, None)
                solver.highs.startCallback(callback_type)

        return res, internal_options

    def _solver_arguments(
            self,
            runner: Runner,
            solver: str,
            model: cp.Model,
            seed: Optional[int] = None,
            intermediate: bool = False,
            cores: int = 1,
            mem_limit: Optional[int] = None,
            **kwargs
        ):
        opt = model.has_objective()
        sat = not opt

        if solver == "ortools":
            return self._ortools_arguments(runner, model, cores=cores, seed=seed, intermediate=intermediate, **kwargs)
        elif solver == "exact":
            return self._exact_arguments(seed=seed, **kwargs)
        elif solver == "choco":
            return self._choco_arguments()
        elif solver == "z3":
            return self._z3_arguments(model, cores=cores, seed=seed, mem_limit=mem_limit, **kwargs)
        elif solver.startswith("minizinc"):  # also can have a subsolver
            return self._minizinc_arguments(solver, cores=cores, seed=seed, **kwargs)
        elif solver == "gurobi":
            return self._gurobi_arguments(runner, model, cores=cores, seed=seed, mem_limit=mem_limit, intermediate=intermediate, opt=opt, **kwargs)
        elif solver == "cpo":
            return self._cpo_arguments(runner, model=model, cores=cores, seed=seed, intermediate=intermediate, **kwargs)
        elif solver == "hexaly":
            return self._hexaly_arguments(runner, model, cores=cores, seed=seed, intermediate=intermediate, **kwargs)
        elif solver == "cplex":
            return self._cplex_arguments(cores=cores, **kwargs)
        elif solver == "highs":
            return self._highs_arguments(runner, model, cores=cores, seed=seed, intermediate=intermediate, **kwargs)
        else:
            runner.print_comment(f"setting parameters of {solver} is not (yet) supported")
            return dict(), None

    def participate_solver_args(self, runner: Runner, solver_args: dict):
        intermediate = self.intermediate and (runner.model is not None and runner.model.has_objective())
        args, internal_options = self._solver_arguments(runner, runner.solver, model=runner.model, seed=self.seed,
                                        intermediate=intermediate,
                                        cores=self.cores, mem_limit=_mib_as_bytes(self.mem_limit) if self.mem_limit is not None else None,
                                        **self.kwargs)

        if internal_options is not None:
            internal_options(runner.s)
        solver_args |= args
        runner.print_comment(f"Solver arguments: {args}")
