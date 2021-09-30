#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## minizinc.py
##
"""
    Interface to the python 'minizinc' package

    Requires that the 'minizinc' python package is installed:

        $ pip install minizinc
    
    as well as the MiniZinc bundled binary packages, downloadable from:
    https://www.minizinc.org/software.html

    Note for Jupyter users: MiniZinc uses AsyncIO, so using it in a jupyter notebook gives
    you the following error: RuntimeError: asyncio.run() cannot be called from a running event loop
    You can overcome this by `pip install nest_asyncio`
    and adding in the top cell `import nest_asyncio; nest_asyncio.apply()`

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

import numpy as np
from datetime import timedelta # for mzn's timeout
from .solver_interface import SolverInterface, ExitStatus, SolverStatus
from ..transformations.get_variables import get_variables_model
from ..expressions.variables import _NumVarImpl, _IntVarImpl, _BoolVarImpl, NegBoolView
from ..expressions.core import Expression, Comparison, Operator
from ..expressions.utils import is_num, is_any_list

class CPM_minizinc(SolverInterface):
    """
    Creates the following attributes:

    user_vars: variables used in the model (without auxiliaries),
               these variables' .value() will be backpopulated on solve
    mzn_solve: the minizinc.Solver instance
    mzn_model: the minizinc.Model instance
    mzn_txt_solve: the 'solve' item in text form, so it can be overwritten
    """

    @staticmethod
    def supported():
        """
            Make sure you installed the minizinc distribution from minizinc.org
            as well as installing the 'minizinc-python' package (e.g. pip install minizinc)
        """
        try:
            import minizinc
            return True
        except ImportError as e:
            return False

    @staticmethod
    def solvernames():
        """
            Returns solvers supported by MiniZinc on your system

            WARNING, some of them may not actually be installed on your system
            (namely cplex, gurobi, scip, xpress)
            the following are bundled in the bundle: chuffed, coin-bc, gecode
        """
        import minizinc
        import json
        # from minizinc.Solver.lookup()
        out = minizinc.default_driver.run(["--solvers-json"])
        out_lst = json.loads(out.stdout)

        solvers = []
        for s in out_lst:
            name = s["id"].split(".")[-1]
            if name not in ['findmus', 'gist', 'globalizer']: # not actually solvers
                solvers.append(name)
        return solvers


    def __init__(self, cpm_model=None, solver=None):
        """
        Constructor of the solver object

        Requires a CPMpy model as input, and will create the corresponding
        minizinc model and solver object (mzn_model and mzn_solver)

        solver has to be one of solvernames() [str, default: None]
        """
        if not self.supported():
            raise Exception("Install the python 'minizinc-python' package to use this '{}' solver interface".format(self.name))
        import minizinc

        super().__init__()

        solvername = solver
        if solvername is None:
            # default solver
            solvername = "gecode"
        elif solvername.startswith('minizinc:'):
            # strip prepended 'minizinc:'
            solvername = solvername[9:]
        self.name = "minizinc:"+solvername

        import minizinc
        self.mzn_solver = minizinc.Solver.lookup(solvername)

        # minizinc-python API
        # Create a MiniZinc model
        self.mzn_model = minizinc.Model()
        if cpm_model is None:
            self.user_vars = []
            self.mzn_txt_solve = "solve satisfy;"
        else:
            # store original vars and objective (before flattening)
            self.user_vars = get_variables_model(cpm_model)
            (mzn_txt, self.mzn_txt_solve) = self.make_model(cpm_model)
            self.mzn_model.add_string(mzn_txt)
            # do NOT add self.mzn_txt_solve yet, so that it can be overwritten later


    def solve(self, time_limit=None, **kwargs):
        """
            Create and call an Instance with the already created mzn_model and mzn_solver

            Arguments:
            - time_limit:  maximum solve time in seconds (float, optional)

            Additional keyword arguments:
            The minizinc solver parameters are partly defined in its API:
            https://minizinc-python.readthedocs.io/en/latest/api.html#minizinc.instance.Instance.solve

            You can use any of these parameters as keyword argument to `solve()` and they will
            be forwarded to the solver. Examples include:
                - free_search=True              Allow the solver to ignore the search definition within the instance. (Only available when the -f flag is supported by the solver). (Default: 0)
                - optimisation_level=0          Set the MiniZinc compiler optimisation level. (Default: 1; 0=none, 1=single pass, 2=double pass, 3=root node prop, 4,5=probing)
                - all_solutions=True            Computes all solutions. WARNING CPMpy only gives you access to the values of the last solution... so not very useful.
                - ...                           I am not sure where solver-specific arguments are documented, but the docs say that command line arguments can be passed by ommitting the '-' (e.g. 'f' instead of '-f')?

            example:
            o.solve(free_search=True, optimisation_level=0)

            Does not store the minizinc.Instance() or minizinc.Result() (can be deleted)
        """
        import minizinc

        # set time limit?
        if time_limit is not None:
            kwargs['timeout'] = timedelta(seconds=time_limit)

        # hack, we need to add the objective in a way that it can be changed
        # later, so make copy of the mzn_model
        copy_model = self.mzn_model.__copy__() # it is implemented
        copy_model.add_string(self.mzn_txt_solve)
        # Transform Model into an instance
        mzn_inst = minizinc.Instance(self.mzn_solver, copy_model)

        # Solve the instance
        kwargs['output-time'] = True # required for time getting
        mzn_result = mzn_inst.solve(**kwargs)#all_solutions=True)

        mzn_status = mzn_result.status

        # translate status
        self.cpm_status = SolverStatus(self.name)
        if mzn_status == minizinc.result.Status.SATISFIED:
            self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif mzn_status == minizinc.result.Status.ALL_SOLUTIONS:
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
            self.cpm_status.exitstatus = ExitStatus.ERROR
        else:
            raise NotImplementedError # a new status type was introduced, please report on github

        # translate runtime
        if 'time' in mzn_result.statistics:
            self.cpm_status.runtime = mzn_result.statistics['time'] # --output-time

        # translate solution values (of original vars only)
        objective_value = None
        if mzn_status.has_solution():
            # runtime
            mznsol = mzn_result.solution
            if is_any_list(mznsol):
                print("Warning: multiple solutions found, only returning last one")
                mznsol = mznsol[-1]
            self.cpm_status.runtime = mzn_result.statistics['time'].total_seconds()

            # fill in variables
            for var in self.user_vars:
                varname = self.clean_varname(var.name)
                if hasattr(mznsol, varname):
                    var._value = getattr(mznsol, varname)
                else:
                    print("Warning, no value for ",varname)

            # translate objective (if any, otherwise None)
            objective_value = mzn_result.objective

        return self._solve_return(self.cpm_status, objective_value)

    def __add__(self, cons):
        """
            Add an additional (list of) constraints to the model
        """
        if not is_any_list(cons):
            cons = [cons]

        txt_cons = ""
        for con in cons:
            txt_cons += "constraint {};\n".format(self.convert_expression(con))

        self.mzn_model.add_string(txt_cons)
        return self

    def minimize(self, expr):
        """
            Minimize the given objective function

            `minimize()` can be called multiple times, only the last one is stored
        """
        # do not add it to the model, support only one 'solve' entry
        self.mzn_txt_solve = "solve minimize {};\n".format(self.convert_expression(expr))

    def maximize(self, expr):
        """
            Maximize the given objective function

            `maximize()` can be called multiple times, only the last one is stored
        """
        # do not add it to the model, support only one 'solve' entry
        self.mzn_txt_solve = "solve maximize {};\n".format(self.convert_expression(expr))

    def clean_varname(self, varname):
        return varname.replace(',','_').replace('.','_').replace(' ','_').replace('[','_').replace(']','')

    def make_model(self, cpm_model):
        """
            Makes the MiniZinc text formulation out of a CPMpy model

            We do not do any flattening, but try to express the expressions (with subexpressions) directly.

            Luckily, the textual output of many operators etc is close to the input minizinc expects
            (well, maybe that is not a coincidence)

            Typically only needed for internal use, or if you want to inspect the generated minizinc text

            returns (txt_model, txt_objective)
            txt_objective separate (you can just concatenate it), so that we can change it later
            (the minizinc API does not support changing it later natively)
        """
        txt_vars = "% Generated by CPMpy\ninclude \"globals.mzn\";\n\n"
        for var in get_variables_model(cpm_model):
            if isinstance(var, _BoolVarImpl):
                txt_vars += "var bool: {};\n".format(self.clean_varname(var.name))
            elif isinstance(var, _IntVarImpl):
                txt_vars += "var {}..{}: {};\n".format(var.lb, var.ub, self.clean_varname(var.name))

        # we can't unpack lists in convert_expression, so must do it upfront
        # and can't make assumptions on '.flat' existing either...
        # this is dirty code that should not be reused, keeping it hidden in this function...
        def flatlist(lst):
            flatlst = []
            for elem in lst:
                if is_any_list(elem):
                    flatlst += flatlist(elem)
                else:
                    flatlst.append(elem)
            return flatlst

        txt_cons = ""
        for con in flatlist(cpm_model.constraints):
            txt_cons += f"constraint {self.convert_expression(con)};\n"

        txt_obj = "solve satisfy;"
        if cpm_model.objective is not None:
            if cpm_model.objective_max:
                txt_obj = "solve maximize "
            else:
                txt_obj = "solve minimize "
            txt_obj += "{};\n".format(self.convert_expression(cpm_model.objective))
                
        return (txt_vars+"\n"+txt_cons, txt_obj)

    def convert_expression(self, expr):
        """
            Convert a CPMpy expression into a minizinc-compatible string

            recursive: also converts nested subexpressions
        """
        if is_any_list(expr):
            if isinstance(expr, np.ndarray):
                # must flatten
                expr_str = [self.convert_expression(e) for e in expr.flat]
            else:
                expr_str = [self.convert_expression(e) for e in expr]
            if len(expr_str) == 1:
                # unary special case, don't put in list
                return expr_str[0]
            else:
                return "[{}]".format(",".join(expr_str))

        if not isinstance(expr, Expression) or \
           isinstance(expr, _NumVarImpl):
            if expr is True:
                return "true"
            if expr is False:
                return "false"
            # default
            if isinstance(expr, NegBoolView):
                return "not "+self.clean_varname(str(expr._bv))
            return self.clean_varname(str(expr))

        # table(vars, tbl): no [] nesting of args, and special table output...
        if expr.name == "table":
            str_vars = self.convert_expression(expr.args[0])
            str_tbl = "[|\n" # opening
            for row in expr.args[1]:
                str_tbl += ",".join(map(str,row)) + " |" # rows
            str_tbl += "\n|]" # closing
            return "table({}, {})".format(str_vars, str_tbl)
        
        args_str = [self.convert_expression(e) for e in expr.args]

        # standard expressions: comparison, operator, element
        if isinstance(expr, Comparison):
            # pretty printing: add () if nested comp/op
            for e in expr.args:
                if isinstance(e, (Comparison,Operator)):
                    for i in [0,1]:
                        args_str[i] = "({})".format(args_str[i])
            # infix notation
            return "{} {} {}".format(args_str[0], expr.name, args_str[1])

        elif isinstance(expr, Operator):
            # some names differently (the infix names!)
            printmap = {'and': '/\\', 'or': '\\/',
                        'sum': '+', 'sub': '-',
                        'mul': '*', 'div': '/', 'pow': '^'}
            op_str = expr.name
            if op_str in printmap:
                op_str = printmap[op_str]

            # TODO: pretty printing of () as in Operator?

            # special case: unary -
            if self.name == '-':
                return "-{}".format(args_str[0])

            # special case, infix: two args
            if len(args_str) == 2:
                for i,arg_str in enumerate(args_str):
                    if isinstance(expr.args[i], Expression):
                        args_str[i] = "("+args_str[i]+")"
                return "{} {} {}".format(args_str[0], op_str, args_str[1])

            # special case: n-ary (non-binary): rename operator
            printnary = {'and': 'forall', 'or': 'exists', 'xor': 'xorall', 'sum': 'sum'}
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
            # minizinc is offset 1, which can be problematic for element...
            idx = args_str[1]
            if isinstance(expr.args[1], _IntVarImpl) and expr.args[1].lb == 0:
                idx = "{}+1".format(idx)
            # almost there
            txt  = "\n    let {{ array[int] of var {}: arr={} }} in\n".format(subtype, args_str[0])
            txt += f"      arr[{idx}]"
            return txt
        
        # rest: global constraints
        elif expr.name.endswith('circuit'): # circuit, subcircuit
            # minizinc is offset 1, which can be problematic here...
            if any(isinstance(e, _IntVarImpl) and e.lb == 0 for e in expr.args):
                # redo args_str[0]
                args_str = ["{}+1".format(self.convert_expression(e)) for e in expr.args]
        
        # default (incl name-compatible global constraints...)
        return "{}([{}])".format(expr.name, ",".join(args_str))
