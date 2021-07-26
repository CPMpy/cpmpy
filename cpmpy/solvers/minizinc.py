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

from .solver_interface import SolverInterface, ExitStatus, SolverStatus
from ..transformations.get_variables import get_variables_model
from ..expressions.variables import _NumVarImpl, _IntVarImpl, _BoolVarImpl, NegBoolView
from ..expressions.core import Expression, Comparison, Operator
from ..expressions.utils import is_num, is_any_list

class CPM_minizinc(SolverInterface):
    """
    Creates the following attributes:

    mzn_inst: the minizinc.Instance created by _model()
    mzn_result: the minizinc.Result used in solve()
    """

    @staticmethod
    def supported():
        try:
            import minizinc
            return True
        except ImportError as e:
            return False

    def __init__(self, cpm_model=None, solvername=None):
        """
        Constructor of the solver object

        Requires a CPMpy model as input, and will create the corresponding
        minizinc model and solver object (mzn_model and mzn_solver)
        """
        if not self.supported():
            raise Exception("Install the python 'minizinc-python' package to use this '{}' solver interface".format(self.name))
        import minizinc

        super().__init__()
        self.name = "minizinc_python"

        import minizinc
        if solvername is None:
            # default solver
            solvername = "gecode"
        self.mzn_solver = minizinc.Solver.lookup(solvername)

        # minizinc-python API
        # Create a MiniZinc model
        self.mzn_model = minizinc.Model()
        if cpm_model is None:
            self.user_vars = []
        else:
            # store original vars and objective (before flattening)
            self.user_vars = get_variables_model(cpm_model)
            mzn_txt = self.make_model(cpm_model)
            self.mzn_model.add_string(mzn_txt)


    def solve(self, **kwargs):
        """
            Call the (already created) solver

            keyword arguments can be any argument accepted by minizinc.Instance.solve()
            For example, set 'all_solutions=True' to have it enumerate all solutions
        """
        import minizinc

        # Transform Model into an instance
        self.mzn_inst = minizinc.Instance(self.mzn_solver, self.mzn_model)

        # Solve the instance
        kwargs['output-time'] = True # required for time getting
        self.mzn_result = self.mzn_inst.solve(**kwargs)#all_solutions=True)

        mzn_status = self.mzn_result.status

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
        if 'time' in self.mzn_result.statistics:
            self.cpm_status.runtime = self.mzn_result.statistics['time'] # --output-time

        # translate solution values (of original vars only)
        if mzn_status.has_solution():
            # runtime
            mznsol = self.mzn_result.solution
            self.cpm_status.runtime = self.mzn_result.statistics['time'].total_seconds()

            # fill in variables
            for var in self.user_vars:
                varname = self.clean_varname(var.name)
                if hasattr(mznsol, varname):
                    var._value = getattr(mznsol, varname)
                else:
                    print("Warning, no value for ",varname)

            # translate objective (if any, otherwise None)
            objective_value = self.mzn_result.objective

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
        raise NotImplementedError("not yet implemented for CPM_minzinc")
    def maximize(self, expr):
        """
            Maximize the given objective function

            `maximize()` can be called multiple times, only the last one is stored
        """
        raise NotImplementedError("not yet implemented for CPM_minzinc")

    def clean_varname(self, varname):
        return varname.replace('[','_').replace(']','')

    def make_model(self, cpm_model):
        """
            Makes the MiniZinc text formulation out of a CPMpy model

            We do not do any flattening, but try to express the expressions (with subexpressions) directly.

            Luckily, the textual output of many operators etc is close to the input minizinc expects
            (well, maybe that is not a coincidence)

            Typically only needed for internal use, or if you want to inspect the generated minizinc text
        """
        txt_vars = "% Generated by CPMpy\ninclude \"globals.mzn\";\n\n"
        for var in get_variables_model(cpm_model):
            if isinstance(var, _BoolVarImpl):
                txt_vars += "var bool: {};\n".format(self.clean_varname(var.name))
            elif isinstance(var, _IntVarImpl):
                txt_vars += "var {}..{}: {};\n".format(var.lb, var.ub, self.clean_varname(var.name))

        txt_cons = ""
        for con in cpm_model.constraints:
            txt_cons += "constraint {};\n".format(self.convert_expression(con))

        txt_obj = "solve "
        if cpm_model.objective is None:
            txt_obj += "satisfy;"
        else:
            if cpm_model.objective_max:
                txt_obj += "maximize "
            else:
                txt_obj += "minimize "
            txt_obj += "{};\n".format(self.convert_expression(cpm_model.objective))
                
        return txt_vars+"\n"+txt_cons+txt_obj

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
            return "[{}]".format(",".join(expr_str))

        if not isinstance(expr, Expression) or \
           isinstance(expr, _NumVarImpl):
            if expr is True:
                return "true"
            if expr is False:
                return "false"
            # default
            return self.clean_varname(str(expr))
        
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
            if all(v.is_bool() for v in expr.args[0]):
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
