#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## minizinc.py
##
"""
    Interface to MiniZinc's Python API.

    MiniZinc is a free and open-source constraint modeling language.
    MiniZinc is used to model constraint satisfaction and optimization problems in
    a high-level, solver-independent way, taking advantage of a large library of
    pre-defined constraints. The model is then compiled into FlatZinc, a solver input
    language that is understood by a wide range of solvers.
    https://www.minizinc.org

    The MiniZinc interface is text-based: CPMpy writes a textfile and passes it to the minizinc Python package.

    Always use :func:`cp.SolverLookup.get("minizinc") <cpmpy.solvers.utils.SolverLookup.get>` to instantiate the solver object.

    ============
    Installation
    ============

    Requires that the 'minizinc' python package is installed:

    .. code-block:: console

        $ pip install minizinc

    as well as the MiniZinc bundled binary packages, downloadable from:
    https://www.minizinc.org/software.html

    See detailed installation instructions at:
    https://minizinc-python.readthedocs.io/en/latest/getting_started.html

    Note for **Jupyter notebook** users: MiniZinc uses AsyncIO, so using it in a Jupyter notebook gives
    you the following error: ``RuntimeError: asyncio.run() cannot be called from a running event loop``
    You can overcome this by ``pip install nest_asyncio``
    and adding in the top cell ``import nest_asyncio; nest_asyncio.apply()``

    The rest of this documentation is for advanced users.

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_minizinc

    ==============
    Module details
    ==============
"""
import re
from typing import Optional
import warnings
import sys
import os
import json
from datetime import timedelta  # for mzn's timeout

import numpy as np

from .solver_interface import SolverInterface, SolverStatus, ExitStatus, Callback
from ..exceptions import MinizincNameException, MinizincBoundsException
from ..expressions.core import Expression, Comparison, Operator, BoolVal
from ..expressions.python_builtins import any as cpm_any
from ..expressions.variables import _NumVarImpl, _IntVarImpl, _BoolVarImpl, NegBoolView, cpm_array
from ..expressions.globalconstraints import DirectConstraint
from ..expressions.utils import is_num, is_any_list, argvals, argval, get_nonneg_args
from ..transformations.decompose_global import decompose_in_tree, decompose_objective
from ..exceptions import MinizincPathException, NotSupportedError
from ..transformations.get_variables import get_variables
from ..transformations.normalize import toplevel_list


class CPM_minizinc(SolverInterface):
    """
    Interface to MiniZinc's Python API

    Creates the following attributes (see parent constructor for more):

    - ``mzn_model``: object, the minizinc.Model instance
    - ``mzn_solver``: object, the minizinc.Solver instance
    - ``mzn_txt_solve``: str, the 'solve' item in text form, so it can be overwritten
    - ``mzn_result``: object, containing solve results

    The :class:`~cpmpy.expressions.globalconstraints.DirectConstraint`, when used, adds a constraint with that name and the given args to the MiniZinc model.

    Documentation of the solver's own Python API:
    https://minizinc-python.readthedocs.io/
    """

    supported_global_constraints = frozenset({"alldifferent", "alldifferent_except0", "allequal",
                                              "inverse", "ite", "xor", "table", "InDomain", "negative_table", "cumulative", "circuit", "gcc",
                                              "increasing", "decreasing",
                                              "strictly_increasing", "strictly_decreasing", "lex_lesseq", "lex_less",
                                              "lex_chain_less","lex_chain_lesseq",
                                              "precedence", "no_overlap",
                                              "min", "max", "abs", "div", "mod", "pow", "element", "count", "nvalue", "among"})
    supported_reified_global_constraints = supported_global_constraints - {"circuit", "precedence"}

    required_version = (2, 8, 2)

    @staticmethod
    def supported(): 
        return CPM_minizinc.installed() and CPM_minizinc.executable_installed() and not CPM_minizinc.outdated()

    @staticmethod
    def installed():
        # try to import the package
        try:
            #  check if MiniZinc Python is installed
            import minizinc
            return True
        except ModuleNotFoundError:
            return False
        except Exception as e:
            raise e

    @staticmethod
    def executable_installed():
        # check if MiniZinc executable is installed
        from minizinc import default_driver
        if default_driver is None:
            warnings.warn("MiniZinc Python is installed, but the MiniZinc executable is missing in path.")
            return False
        return True

    @staticmethod
    def outdated():
        from minizinc import default_driver
        if default_driver.parsed_version >= CPM_minizinc.required_version:
            return False
        else:
            # outdated
            return True
                

    @staticmethod
    def solvernames(installed:bool=True, with_version:bool=False):
        """
            Returns solvers supported by MiniZinc (on your system) with optionally their installed version.

            Arguments:
                installed (boolean): whether to filter the solvernames to those installed on your system (default True)
                with_version (boolean): whether to additionally return the available version matching with each solvername 
                                        (if not available on the system, the entry defaults to None)

            Returns:
                list of solver names if with_version==False, otherwise a tuple of two lists: the solver names and their versions

            .. warning::
                WARNING, some of the returned solver names (when ``installed=False``) may not actually 
                be installed on your system (namely cplex, gurobi, scip, xpress).
                The following are bundled with minizinc: chuffed, coin-bc, gecode.
                Use ``installed=True`` (the default) if you only want the names of the actually installed solvers.
        """

        # Collect solver names and versions
        if CPM_minizinc.supported(): # check if minizinc is installed
            import minizinc
            driver = minizinc.default_driver
            
            # Collect solver names
            all_solvers, all_versions = [], []
            output = driver._run(["--solvers-json"]) # get json-structured solver overview
            solvers = json.loads(output.stdout)        
            for solver_dict in solvers:
                # get subsolver metadata
                tag = solver_dict["id"].split(".")[-1]
                version = solver_dict["version"]
                if tag not in ['findmus', 'gist', 'globalizer']: # some are not actually solvers
                    if tag not in all_solvers:
                        all_solvers.append(tag)
                        if version == '<unknown version>': # if no version info available, default to None
                            version = None
                        all_versions.append(version)
        else:
            warnings.warn("MiniZinc is not installed or not supported on this system.")
            if with_version:
                return ([], [])
            else:
                return []
            

        if not installed:
            """
            Return all solver names, without checking if they're actually available on the system.
            """
            if with_version:
                return (all_solvers, all_versions)
            else:
                return all_solvers

        else:
            """
            Test which solver executables are available by retrieving version information
            """
            valid_indices = [i for i, version in enumerate(all_versions) if version != "<unknown version>"] # check if version is available (required by minizinc)
            installed_solvers = [all_solvers[i] for i in valid_indices]
            installed_versions = [all_versions[i] for i in valid_indices]

            if with_version:
                return (installed_solvers, installed_versions)
            else:
                return installed_solvers       
             
    @staticmethod
    def solverversion(subsolver:str) -> Optional[str]:
        """
        Returns the version of the requested subsolver.

        Arguments:
            subsolver (str): name of the subsolver

        Returns:
            Version number of the subsolver if installed, else None 
        """
        all_solvers, all_versions = CPM_minizinc.solvernames(installed=False, with_version=True)
        try:
            solver_index = all_solvers.index(subsolver) # find requested subsolver
            return all_versions[solver_index]
        except ValueError:
            raise ValueError(f"Subsolver '{subsolver}' not found in the list of available solvers.")

    @staticmethod
    def version() -> Optional[str]:
        """
        Returns the installed version of the solver's Python API.

        For Minizinc, two version numbers get returned: ``<minizinc python API version>/<minizinc driver version>``
        """
        from importlib.metadata import version, PackageNotFoundError
        try:
            from minizinc import default_driver
            mzn_version = version("minizinc")
            solver_version = '.'.join(str(a) for a in default_driver.parsed_version)
            return f"{mzn_version}/{solver_version}"
        except (PackageNotFoundError, ModuleNotFoundError):
            return None

    # variable name can not be any of these keywords
    keywords = frozenset(['ann', 'annotation', 'any', 'array', 'bool', 'case', 'constraint', 'diff', 'div', 'else',
                          'elseif', 'endif', 'enum', 'false', 'float', 'function', 'if', 'in', 'include', 'int',
                          'intersect', 'let', 'list', 'maximize', 'minimize', 'mod', 'not', 'of', 'op', 'opt', 'output',
                          'par', 'predicate', 'record', 'satisfy', 'set', 'solve', 'string', 'subset', 'superset',
                          'symdiff', 'test', 'then', 'true', 'tuple', 'type', 'union', 'var', 'where', 'xor'])
    # variable names must have this pattern
    mzn_name_pattern = re.compile('^[A-Za-z][A-Za-z0-9_]*$')

    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Arguments:
            cpm_model: Model(), a CPMpy Model() (optional)
            subsolver: str, name of a subsolver (optional)
                          has to be one of solvernames()
        """
        if not self.installed():
            raise ModuleNotFoundError("CPM_minizinc: Install the python package 'cpmpy[minizinc]' to use this solver interface.")
        elif not self.executable_installed():
            raise ModuleNotFoundError("CPM_minizinc: Install the MiniZinc executable and make it available in path.")
        elif self.outdated():
            version = str(self.required_version[0])
            for x in self.required_version[1:]:
                version = version + "." + str(x)
            raise ImportError("Your Minizinc compiler is outdated, please upgrade to a version >= " + version)

        import minizinc

        # determine subsolver
        if subsolver is None or subsolver == 'minizinc':
            # default solver
            subsolver = "gecode"
        elif subsolver.startswith('minizinc:'):
            subsolver = subsolver[9:]  # strip 'minizinc:'

        # initialise the native solver object
        # (so its params can still be changed before calling solve)
        self.mzn_solver = minizinc.Solver.lookup(subsolver)
        # It is the model object that contains the constraints for minizinc
        self.mzn_model = minizinc.Model()
        self.mzn_model.add_string("% Generated by CPMpy\ninclude \"globals.mzn\";\n\n")
        # Prepare solve statement, so it can be overwritten on demand
        self.mzn_txt_solve = "solve satisfy;"
        self.mzn_result = None

        # initialise everything else and post the constraints/objective
        super().__init__(name="minizinc:"+subsolver, cpm_model=cpm_model)

    @property
    def native_model(self):
        """
            Returns the solver's underlying native model (for direct solver access).
        """
        return self.mzn_model


    def _pre_solve(self, time_limit:Optional[float]=None, **kwargs):
        """ shared by solve() and solveAll() """
        import minizinc

        if time_limit is not None:
            kwargs['timeout'] = timedelta(seconds=time_limit)

        # hack, we need to add the objective in a way that it can be changed
        # later, so make copy of the mzn_model
        copy_model = self.mzn_model.__copy__()  # it is implemented
        copy_model.add_string(self.mzn_txt_solve)
        # Transform Model into an instance
        mzn_inst = minizinc.Instance(self.mzn_solver, copy_model)

        kwargs['output-time'] = True  # required for time getting
        return (kwargs, mzn_inst)

    def solve(self, time_limit:Optional[float]=None, **kwargs):
        """
            Call the MiniZinc solver
            
            Creates and calls an Instance with the already created ``mzn_model`` and ``mzn_solver``

            Arguments:
                time_limit (float, optional):       maximum solve time in seconds 
                **kwargs (any keyword argument):    sets parameters of solver object
                
            
            Arguments that correspond to solver parameters:

            =======================  ===========
            Keyword                  Description
            =======================  ===========
            free_search=True              Allow the solver to ignore the search definition within the instance. (Only available when the -f flag is supported by the solver). (Default: 0)
            optimisation_level=0          Set the MiniZinc compiler optimisation level. (Default: 1; 0=none, 1=single pass, 2=double pass, 3=root node prop, 4,5=probing)
            =======================  ===========             
            
            
            I am not sure where solver-specific arguments are documented, but the docs say that command line arguments can be passed by ommitting the '-' (e.g. 'f' instead of '-f')?
            
            The minizinc solver parameters are partly defined in its API:
            https://minizinc-python.readthedocs.io/en/latest/api.html#minizinc.instance.Instance.solve

            Does not store the ``minizinc.Instance()`` or ``minizinc.Result()``
        """

        if time_limit is not None and time_limit <= 0:
            raise ValueError("Time limit must be positive")

        # ensure all vars are known to solver
        self.solver_vars(list(self.user_vars))

        # make mzn_inst
        (mzn_kwargs, mzn_inst) = self._pre_solve(time_limit=time_limit, **kwargs)
        
        # call the solver, with parameters
        import minizinc.error
        try:
            self.mzn_result = mzn_inst.solve(**mzn_kwargs)
        except minizinc.error.MiniZincError as e:
            if sys.platform == "win32" or sys.platform == "cygwin":  # path error can occur in windows
                path = os.environ.get("path")
                if "MiniZinc" in str(path):
                    warnings.warn('You might have the wrong minizinc PATH set (windows user Environment Variables')
                    raise e
                else:
                    raise MinizincPathException("Please add your minizinc installation folder to the user Environment PATH variable")
            else:
                raise e
        # new status, translate runtime
        self.cpm_status = self._post_solve(self.mzn_result)

        # True/False depending on self.cpm_status
        has_sol = self._solve_return(self.cpm_status)

        # translate solution values (of user specified variables only)
        self.objective_value_ = None
        if has_sol:  # mzn_result.status.has_solution():
            mznsol = self.mzn_result.solution
            if is_any_list(mznsol):
                print("Warning: multiple solutions found, only returning last one")
                mznsol = mznsol[-1]

            # fill in variable values
            for cpm_var in self.user_vars:
                sol_var = self.solver_var(cpm_var)
                if hasattr(mznsol, sol_var):
                    cpm_var._value = getattr(mznsol, sol_var)
                else:
                    raise ValueError(f"Var {cpm_var} is unknown to the Minizinc solver, this is unexpected - please report on github...")

            # translate objective, for optimisation problems only (otherwise None)
            self.objective_value_ = self.mzn_result.objective

        else: # clear values of variables
            for cpm_var in self.user_vars:
                cpm_var._value = None

        return has_sol

    def _post_solve(self, mzn_result, solve_all:bool=False):
        """ shared by solve() and solveAll() """
        import minizinc

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        runtime = 0
        if 'time' in mzn_result.statistics:
            self.cpm_status.runtime = self.mzn_time_to_seconds(mzn_result.statistics.get("time"))
        else:
            runtime += self.mzn_time_to_seconds(mzn_result.statistics.get("flatTime", 0))
            runtime += self.mzn_time_to_seconds(mzn_result.statistics.get("initTime", 0))
            runtime += self.mzn_time_to_seconds(mzn_result.statistics.get("solveTime", 0))
            if runtime != 0:
                self.cpm_status.runtime = runtime
            else:
                raise NotImplementedError  # Please report on github, minizinc probably changed their time names/types

        # translate exit status
        mzn_status = mzn_result.status
        if mzn_status == minizinc.result.Status.SATISFIED:
            self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif mzn_status == minizinc.result.Status.ALL_SOLUTIONS:
            if solve_all:
                self.cpm_status.exitstatus = ExitStatus.OPTIMAL
            else:
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif mzn_status == minizinc.result.Status.OPTIMAL_SOLUTION:
            self.cpm_status.exitstatus = ExitStatus.OPTIMAL
        elif mzn_status == minizinc.result.Status.UNSATISFIABLE:
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        elif mzn_status == minizinc.result.Status.ERROR:
            self.cpm_status.exitstatus = ExitStatus.ERROR
            raise Exception("MiniZinc solver returned with status 'Error'")
        elif mzn_status == minizinc.result.Status.UNKNOWN:
            # means, no solution was found (e.g. within timeout?)...
            self.cpm_status.exitstatus = ExitStatus.UNKNOWN
        else:
            raise NotImplementedError  # a new status type was introduced, please report on github

        return self.cpm_status

    def mzn_time_to_seconds(self, time):
        if isinstance(time, int):
            return time / 1000
        elif isinstance(time, timedelta):
            return time.total_seconds()  # --output-time
        else:
            raise NotImplementedError  # unexpected type for time

    async def _solveAll(self, display=None, time_limit:Optional[float]=None, solution_limit:Optional[int]=None, **kwargs):
        """ Special 'async' function because mzn.solutions() is async """

        # ensure all vars are known to solver
        self.solver_vars(list(self.user_vars))
        
        # make mzn_inst
        (kwargs, mzn_inst) = self._pre_solve(time_limit=time_limit, **kwargs)
        kwargs['all_solutions'] = True

        solution_count = 0
        # has an asynchronous generator
        async for mzn_result in mzn_inst.solutions(**kwargs):
            # was the last one
            if mzn_result.solution is None:
                break

             # fill in variable values
            mznsol = mzn_result.solution
            for cpm_var in self.user_vars:
                sol_var = self.solver_var(cpm_var)
                if hasattr(mznsol, sol_var):
                    cpm_var._value = getattr(mznsol, sol_var)
                else:
                    raise ValueError(f"Var {cpm_var} is unknown to the Minizinc solver, this is unexpected - please report on github...")

            # display if needed
            if display is not None:
                if isinstance(display, Expression):
                    print(argval(display))
                elif isinstance(display, list):
                    print(argvals(display))
                else:
                    display()  # callback

            # count and stop
            solution_count += 1
            if solution_count == solution_limit:
                break

            # add nogood on the user variables
            self += cpm_any([v != v.value() for v in self.user_vars])

        if solution_count == 0:
            # clear user vars if no solution found
            self.objective_value_ = None
            for var in self.user_vars:
                var._value = None

        # status handling
        self._post_solve(mzn_result, solve_all=True)

        if solution_count: # found at least one solution
            if solution_count == solution_limit: # matched solution limit
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
            # elif mzn_result.solution is None: <- is implicit since nothing needs to update
                # last iteration didn't find a solution
                # nothing needs to update since _post_solve already set state correctly (state from the second-last iteration)

        return solution_count

    def solver_var(self, cpm_var) -> str:
        """
            Creates solver variable for cpmpy variable
            or returns from cache if previously created.

            Returns:
                minizinc-friendly 'string' name of var.

            .. warning::
                WARNING, this assumes it is never given a 'NegBoolView'
                might not be true... e.g. in revar after solve?
        """
        if is_num(cpm_var):
            if cpm_var < -2147483646 or cpm_var > 2147483646:
                raise MinizincBoundsException(
                    "minizinc does not accept integer literals with bounds outside of range (-2147483646..2147483646)")
            return str(cpm_var)

        # Assumes it is never given a 'NegBoolView'
        if isinstance(cpm_var, NegBoolView):
            raise NotSupportedError("Negative literals are not handled here. Please report.")

        if cpm_var not in self._varmap:
            # clean the varname
            varname = cpm_var.name
            mzn_var = varname.replace(',', '_').replace('.', '_').replace(' ', '_').replace('[', '_').replace(']', '')

            # test if the name is a valid minizinc identifier
            if not self.mzn_name_pattern.search(mzn_var):
                raise MinizincNameException("Minizinc only accept names with alphabetic characters, "
                                            "digits and underscores. "
                                "First character must be an alphabetic character")
            if mzn_var in self.keywords:
                raise MinizincNameException(f"This variable name is a disallowed keyword in MiniZinc: {mzn_var}")

            if isinstance(cpm_var, _BoolVarImpl):
                self.mzn_model.add_string(f"var bool: {mzn_var};\n")
            elif isinstance(cpm_var, _IntVarImpl):
                if cpm_var.lb < -2147483646 or cpm_var.ub > 2147483646:
                    raise MinizincBoundsException("minizinc does not accept variables with bounds outside "
                                                  "of range (-2147483646..2147483646)")
                self.mzn_model.add_string(f"var {cpm_var.lb}..{cpm_var.ub}: {mzn_var};\n")
            self._varmap[cpm_var] = mzn_var

        return self._varmap[cpm_var]

    def objective(self, expr, minimize):
        """
            Post the given expression to the solver as objective to minimize/maximize

            - expr: Expression, the CPMpy expression that represents the objective function
            - minimize: Bool, whether it is a minimization problem (True) or maximization problem (False)

            'objective()' can be called multiple times, only the last one is stored
        """

        # save user variables
        get_variables(expr, collect=self.user_vars) # add objvars to vars

        obj, decomp_cons = decompose_objective(expr,
                                               supported=self.supported_global_constraints,
                                               supported_reified=self.supported_reified_global_constraints,
                                               csemap=self._csemap)
        self.add(decomp_cons)

        # make objective function or variable and post

        mzn_obj = self._convert_expression(obj)
        # do not add it to the mzn_model yet, supports only one 'solve' entry
        if minimize:
            self.mzn_txt_solve = "solve minimize {};\n".format(mzn_obj)
        else:
            self.mzn_txt_solve = "solve maximize {};\n".format(mzn_obj)

    def has_objective(self):
        return self.mzn_txt_solve != "solve satisfy;"

    def transform(self, cpm_expr):
        """
            Decompose globals not supported (and flatten globalfunctions)
            ensure it is a list of constraints

            :param cpm_expr: CPMpy expression, or list thereof
            :type cpm_expr: Expression or list of Expression

            :return: list of Expression
        """
        cpm_cons = toplevel_list(cpm_expr)
        cpm_cons = decompose_in_tree(cpm_cons,
                                     supported=self.supported_global_constraints,
                                     supported_reified=self.supported_reified_global_constraints,
                                     csemap=self._csemap)
        return cpm_cons

    def add(self, cpm_expr):
        """
            Translate a CPMpy constraint to MiniZinc string and add it to the solver

            Any CPMpy expression given is immediately transformed (through `transform()`)
            and then posted to the solver in this function.

            This can raise 'NotImplementedError' for any constraint not supported after transformation

            The variables used in expressions given to add are stored as 'user variables'. Those are the only ones
            the user knows and cares about (and will be populated with a value after solve). All other variables
            are auxiliary variables created by transformations.

        :param cpm_expr: CPMpy expression, or list thereof
        :type cpm_expr: Expression or list of Expression

        :return: self
        """
        get_variables(cpm_expr, collect=self.user_vars)
        # transform and post the constraints
        for cpm_con in self.transform(cpm_expr):
            # Get text expression, add to the solver
            mzn_str = f"constraint {self._convert_expression(cpm_con)};\n"
            self.mzn_model.add_string(mzn_str)

        return self
    __add__ = add  # avoid redirect in superclass

    def _convert_expression(self, expr) -> str:
        """
            Convert a CPMpy expression into a minizinc-compatible string

            recursive: also converts nested subexpressions, so we need a
            function that returns strings
        """
        if is_any_list(expr):
            if isinstance(expr, np.ndarray):
                # must flatten
                expr_str = [self._convert_expression(e) for e in expr.flat]
            else:
                expr_str = [self._convert_expression(e) for e in expr]
            return "[{}]".format(",".join(expr_str))

        if isinstance(expr, (bool, np.bool_)):
            expr = BoolVal(expr)

        if not isinstance(expr, Expression):
            return self.solver_var(expr)  # constants

        if isinstance(expr, BoolVal):
            return str(expr.args[0]).lower()

        # default
        if isinstance(expr, _NumVarImpl):
            if isinstance(expr, NegBoolView):
                return "not " + self.solver_var(expr._bv)
            return self.solver_var(expr)

        # table(vars, tbl): no [] nesting of args, and special table output...
        if expr.name == "table":
            str_vars = self._convert_expression(expr.args[0])
            str_tbl = "[|\n"  # opening
            for row in expr.args[1]:
                str_tbl += ",".join(map(str, row)) + " |"  # rows
            str_tbl += "\n|]"  # closing
            return "table({}, {})".format(str_vars, str_tbl)

        # negative_table(vars, tbl): use not table(...) for forbidden assignments
        if expr.name == "negative_table":
            str_vars = self._convert_expression(expr.args[0])
            str_tbl = "[|\n"  # opening
            for row in expr.args[1]:
                str_tbl += ",".join(map(str, row)) + " |"  # rows
            str_tbl += "\n|]"  # closing
            return "not table({}, {})".format(str_vars, str_tbl)

        # inverse(fwd, rev): unpack args and work around MiniZinc's default 1-based indexing
        if expr.name == "inverse":
            def zero_based(array):
                return "array1d(0..{}, {})".format(len(array)-1, self._convert_expression(array))

            str_fwd = zero_based(expr.args[0])
            str_rev = zero_based(expr.args[1])
            return "inverse({}, {})".format(str_fwd, str_rev)

        if expr.name == "alldifferent_except0":
            args_str = [self._convert_expression(e) for e in expr.args]
            return "alldifferent_except_0([{}])".format(",".join(args_str))

        if expr.name in ["lex_lesseq", "lex_less"]:
            X = [self._convert_expression(e) for e in expr.args[0]]
            Y = [self._convert_expression(e) for e in expr.args[1]]
            return f"{expr.name}({{}}, {{}})".format(X, Y)

        if expr.name in ["lex_chain_less", "lex_chain_lesseq"]:
            X = cpm_array([[self._convert_expression(e) for e in row] for row in expr.args])
            str_X = "[|\n"  # opening
            for row in X.T:  # Minizinc enforces lexicographic order on columns
                str_X += ",".join(map(str, row)) + " |"  # rows
            str_X += "\n|]"  # closing
            return f"{expr.name}({{}})".format(str_X)

        elif expr.name == "cumulative":
            start, dur, end, demand, capacity = expr.args

            global_str = "cumulative({},{},{},{})"
            # ensure duration is non-negative
            dur, extra_cons = get_nonneg_args(dur)
            # ensure demand is non-negative
            demand, demand_cons = get_nonneg_args(demand)
            extra_cons += demand_cons

            if end is not None:
                extra_cons += [s + d == e for s, d, e in zip(start, dur, end)]

            format_str = "forall(" + self._convert_expression(extra_cons) + " ++ [" + global_str + "])"

            return format_str.format(self._convert_expression(start),
                                     self._convert_expression(dur),
                                     self._convert_expression(demand),
                                     self._convert_expression(capacity))

        elif expr.name == "no_overlap":
            start, dur, end = expr.args
            global_str = "disjunctive({},{})"
            # ensure duration is non-negative
            dur, extra_cons = get_nonneg_args(dur)
            if end is not None:
                extra_cons += [s + d == e for s, d, e in zip(start, dur, end)]

            format_str = "forall(" + self._convert_expression(extra_cons) + " ++ [" + global_str + "])"

            return format_str.format(self._convert_expression(start), self._convert_expression(dur))

        args_str = [self._convert_expression(e) for e in expr.args]
        # standard expressions: comparison, operator, element
        if isinstance(expr, Comparison):
            # wrap args that are a subexpression in ()
            for i, arg_str in enumerate(args_str):
                if isinstance(expr.args[i], Expression):  # (Comparison, Operator)
                    args_str[i] = "(" + args_str[i] + ")"
            # infix notation
            return "{} {} {}".format(args_str[0], expr.name, args_str[1])

        elif isinstance(expr, Operator):
            # some names differently (the infix names!)
            printmap = {'and': '/\\', 'or': '\\/',
                        'sum': '+', 'sub': '-',
                        'mul': '*'}
            op_str = expr.name
            expr_bounds = expr.get_bounds()
            if expr_bounds[0] < -2147483646 or expr_bounds[1] > 2147483646:
                raise MinizincBoundsException("minizinc does not accept expressions with bounds outside of "
                                              "range (-2147483646..2147483646)")
            if op_str in printmap:
                op_str = printmap[op_str]

            # TODO: pretty printing of () as in Operator?

            # special case: unary -
            if self.name == '-':
                return "-{}".format(args_str[0])

            # very special case: weighted sum (before 2-ary)
            if expr.name == 'wsum':
                # I don't think there is a more direct way unfortunately
                w = [self._convert_expression(wi) for wi in expr.args[0]]
                x = [self._convert_expression(xi) for xi in expr.args[1]]
                args_str = [f"{wi}*({xi})" for wi, xi in zip(w, x)]
                return "{}([{}])".format("sum", ",".join(args_str))

            # special case, infix: two args
            if len(args_str) == 2:
                # wrap args that are a subexpression in ()
                for i, arg_str in enumerate(args_str):
                    if isinstance(expr.args[i], Expression):
                        args_str[i] = "(" + args_str[i] + ")"
                return "{} {} {}".format(args_str[0], op_str, args_str[1])

            # special case: n-ary (non-binary): rename operator
            printnary = {'and': 'forall', 'or': 'exists', 'sum': 'sum'}
            if expr.name in printnary:
                op_str = printnary[expr.name]
                return "{}([{}])".format(op_str, ",".join(args_str))

            # default: prefix printing
            return "{}({})".format(op_str, ",".join(args_str))

        elif expr.name == "element":
            subtype = "int"
            if all(isinstance(v, bool) or \
                   (isinstance(v, Expression) and v.is_bool()) \
                   for v in expr.args[0]):
                subtype = "bool"
            idx = args_str[1]
            # minizinc is offset 1, which can be problematic for element...
            # thx Hakan, fix by using array1d(0..len, []), issue #54
            txt = "\n    let {{ array[int] of var {}: arr=array1d({}..{},{}) }} in\n".format(subtype, 0,
                                                                                             len(expr.args[0]) - 1,
                                                                                             args_str[0])
            txt += f"      arr[{idx}]"
            return txt

        # rest: global constraints
        elif expr.name.endswith('circuit'):  # circuit, subcircuit
            # minizinc is offset 1, which can be problematic here...
            args_str = ["{}+1".format(self._convert_expression(e)) for e in expr.args]

        elif expr.name == "precedence":
            return "value_precede_chain({},{})".format(args_str[1], args_str[0])

        elif expr.name == 'ite':
            cond, tr, fal = expr.args
            return "if {} then {} else {} endif".format(self._convert_expression(cond), self._convert_expression(tr),
                                                        self._convert_expression(fal))

        elif expr.name == "gcc":
            vars, vals, occ = expr.args
            vars = self._convert_expression(vars)
            vals = self._convert_expression(vals)
            occ = self._convert_expression(occ)
            if expr.closed is False:
                name = "global_cardinality"
            else:
                name = "global_cardinality_closed"
            return "{}({},{},{})".format(name, vars, vals, occ)

        elif expr.name == "div":
            return "{} div {}".format(*args_str)

        elif expr.name == "mod":
            return "{} mod {}".format(*args_str)

        elif expr.name == "pow":
            return "{}^{}".format(*args_str)

        elif expr.name == "abs":
            return "abs({})".format(args_str[0])

        elif expr.name == "count":
            vars, val = expr.args
            vars = self._convert_expression(vars)
            val = self._convert_expression(val)
            return "count({},{})".format(vars, val)

        elif expr.name == "among":
            vars, vals = expr.args
            vars = self._convert_expression(vars)
            vals = self._convert_expression(vals).replace("[", "{").replace("]", "}")  # convert to set
            return "among({},{})".format(vars, vals)

        elif expr.name == "InDomain":
            # InDomain(expr, domain_list) - convert domain_list to a set
            expr_str = self._convert_expression(expr.args[0])
            domain = expr.args[1]
            # Convert domain list to set format
            if is_any_list(domain):
                domain_str = "{" + ",".join(self._convert_expression(d) for d in domain) + "}"
            else:
                domain_str = self._convert_expression(domain)
            return "({} in {})".format(expr_str, domain_str)

        elif expr.name == "regular":
            # regular(array, transitions, start, accepting)
            # MiniZinc regular constraint expects: regular(array, transitions_table, start, accepting)
            # where transitions_table is a 2D array
            array, transitions, start, accepting = expr.args
            array_str = self._convert_expression(array)
            # Convert transitions to a 2D array format for MiniZinc
            # transitions is a list of (src, value, dst) tuples
            transitions_list = []
            for src, val, dst in transitions:
                transitions_list.append("[{}, {}, {}]".format(
                    self._convert_expression(src),
                    self._convert_expression(val),
                    self._convert_expression(dst)
                ))
            transitions_str = "[{}]".format(",".join(transitions_list))
            start_str = self._convert_expression(start)
            accepting_str = self._convert_expression(accepting)
            return "regular({}, {}, {}, {})".format(array_str, transitions_str, start_str, accepting_str)

        # a direct constraint, treat differently for MiniZinc, a text-based language
        # use the name as, unpack the arguments from the argument tuple
        elif isinstance(expr, DirectConstraint):
            return "{}({})".format(expr.name, ",".join(args_str))

        print_map = {"allequal": "all_equal", "xor": "xorall"}
        if expr.name in print_map:
            return "{}([{}])".format(print_map[expr.name], ",".join(args_str))

        # default (incl name-compatible global constraints...)
        return "{}([{}])".format(expr.name, ",".join(args_str))

    def solveAll(self, display:Optional[Callback]=None, time_limit:Optional[float]=None, solution_limit:Optional[int]=None, call_from_model=False, **kwargs):
        """
            Compute all solutions and optionally display the solutions.

            MiniZinc-specific implementation.

            Arguments:
                display:            either a list of CPMpy expressions, OR a callback function, called with the variables after value-mapping
                                    default/None: nothing displayed
                time_limit:         stop after this many seconds (default: None)
                solution_limit:     stop after this many solutions (default: None)
                call_from_model:    whether the method is called from a CPMpy Model instance or not
                **kwargs:           any keyword argument, sets parameters of solver object, overwrites construction-time kwargs

            Returns: 
                number of solutions found
        """
        # XXX: check that no objective function??
        if self.has_objective():
            raise NotSupportedError("Minizinc Python does not support finding all optimal solutions (yet)")

        import asyncio

        # set time limit
        if time_limit is not None:
            if time_limit <= 0:
                raise ValueError("Time limit must be positive")

        # HAD TO DEFINE OUR OWN ASYNC HANDLER
        coroutine = self._solveAll(display=display, time_limit=time_limit,
                                    solution_limit=solution_limit, **kwargs)
        # THE FOLLOWING IS STRAIGHT FROM `minizinc.instance.solve()`
        # LETS HOPE IT DOES NOT DIVERGE FROM UPSTREAM
        if sys.version_info >= (3, 7):
            if sys.platform == "win32":
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            return asyncio.run(coroutine)
        else:
            if sys.platform == "win32":
                loop = asyncio.ProactorEventLoop()
            else:
                loop = asyncio.events.new_event_loop()

            try:
                asyncio.events.set_event_loop(loop)
                return loop.run_until_complete(coroutine)
            finally:
                asyncio.events.set_event_loop(None)
                loop.close()

    def minizinc_string(self) -> str:
        """
            Returns the model as represented in the Minizinc language.
        """
        return "".join(self._pre_solve()[1]._code_fragments)

    def flatzinc_string(self, **kwargs) -> str:
        """
            Returns the model as represented in the Flatzinc language.
        """
        with self._pre_solve()[1].flat(**kwargs) as (fzn, ozn, statistics):
            with open(fzn.name) as f:
                f.seek(0)
                contents = f.readlines()
        return "".join(contents)
