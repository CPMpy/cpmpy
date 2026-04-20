#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## scip.py
##
"""
This file implements helper functions for converting CPMpy models to and from various data 
formats supported by the SCIP optimization suite.

============
Installation
============

The 'pyscipopt' python package must be installed separately through `pip`:

.. code-block:: console
    
    $ pip install cpmpy[io.scip]

=================
List of functions
=================

.. autosummary::
    :nosignatures:

    read_scip
    write_scip
    to_scip
"""


import argparse
import math
import os
import sys
import tempfile
import numpy as np
import cpmpy as cp
import warnings

from typing import Union, Optional

from cpmpy.expressions.core import BoolVal, Comparison, Operator
from cpmpy.expressions.variables import _NumVarImpl, _BoolVarImpl, NegBoolView, _IntVarImpl
from cpmpy.transformations.comparison import only_numexpr_equality
from cpmpy.transformations.decompose_global import decompose_in_tree
from cpmpy.transformations.flatten_model import flatten_constraint, flatten_objective
from cpmpy.transformations.get_variables import get_variables
from cpmpy.transformations.linearize import linearize_constraint, only_positive_bv
from cpmpy.transformations.normalize import toplevel_list
from cpmpy.transformations.reification import only_implies, reify_rewrite
from cpmpy.expressions.utils import is_any_list, is_num
from cpmpy.expressions.globalconstraints import DirectConstraint
from cpmpy.expressions.variables import ignore_variable_name_check


_std_open = open
def read_scip(fname: Union[str, os.PathLike], open=open, assume_integer:bool=False) -> cp.Model:
    """
    Read a SCIP-compatible model from a file and return a CPMpy model.

    Arguments:
        fname: The path to the SCIP-compatible file to read.
        open: The function to use to open the file. (SCIP does not require this argument, will be ignored)
        assume_integer: Whether to assume that all variables are integer.

    Returns:
        A CPMpy model.
    """
    if not _SCIPWriter.supported():
        raise Exception("SCIP: Install SCIP IO dependencies: cpmpy[io.scip]")

    with ignore_variable_name_check():
                
        from pyscipopt import Model

        # Load file into pyscipopt model
        scip = Model()
        scip.hideOutput()
        scip.readProblem(filename=fname)
        scip.hideOutput(quiet=False)

        # 1) translate variables
        scip_vars = scip.getVars()
        var_map = {}
        for var in scip_vars:
            name = var.name         # name of the variable
            vtype = var.vtype()     # type of the variable
            if vtype == "BINARY":
                var_map[name] = cp.boolvar(name=name)
            elif vtype == "INTEGER":
                lb = int(var.getLbOriginal())
                ub = int(var.getUbOriginal())
                var_map[name] = cp.intvar(lb, ub, name=name)
            elif vtype == "CONTINUOUS":
                if assume_integer:
                    lb = int(math.ceil(var.getLbOriginal()))
                    ub = int(math.floor(var.getUbOriginal()))
                    if lb != var.getLbOriginal() or ub != var.getUbOriginal():
                        warnings.warn(f"Continuous variable {name} has non-integer bounds {var.getLbOriginal()} - {var.getUbOriginal()}. CPMpy will assume it is integer.")
                    var_map[name] = cp.intvar(lb, ub, name=name)
                else:
                    raise ValueError(f"CPMpy does not support continious variables: {name}")
            else:
                raise ValueError(f"Unsupported variable type: {vtype}")
        

        model = cp.Model()

        # 2) translate constraints
        scip_cons = scip.getConss()
        for cons in scip_cons:
            ctype = cons.getConshdlrName()  # type of the constraint

            if ctype == "linear":
                cons_vars = scip.getConsVars(cons)  # variables in the constraint (x)
                cons_coeff = scip.getConsVals(cons) # coefficients of the variables (A)

                cpm_vars = [var_map[v.name] for v in cons_vars] # convert to CPMpy variables
                cpm_sum = cp.sum(var*coeff for (var,coeff) in zip(cpm_vars, cons_coeff)) # Ax

                lhs = scip.getLhs(cons) # lhs of the constraint
                rhs = scip.getRhs(cons) # rhs of the constraint

                # convert to integer bounds
                _lhs = int(math.ceil(lhs))
                _rhs = int(math.floor(rhs))
                if _lhs != int(lhs) or _rhs != int(rhs):
                    if assume_integer:
                        warnings.warn(f"Constraint {cons.name} has non-integer bounds. CPMpy will assume it is integer.")
                    else:
                        raise ValueError(f"Constraint {cons.name} has non-integer bounds. CPMpy does not support non-integer bounds.")

                # add the constraint to the model
                model += _lhs <= cpm_sum
                model += cpm_sum <= _rhs

            else: 
                raise ValueError(f"Unsupported constraint type: {ctype}")

        # 3) translate objective
        scip_objective = scip.getObjective()
        direction = scip.getObjectiveSense()

        n_terms = len(scip_objective.terms)
        obj_vars = cp.cpm_array([None]*n_terms)
        obj_coeffs = np.zeros(n_terms, dtype=int)

        for i, (term, coeff) in enumerate(scip_objective.terms.items()): # terms is a dictionary mapping terms to coefficients
            if len(term.vartuple) > 1:
                raise ValueError(f"Unsupported objective term: {term}") # TODO <- assumes linear, support higher-order terms
            cpm_var = var_map[term.vartuple[0].name] # TODO <- assumes linear
            obj_vars[i] = cpm_var
            
            _coeff = int(math.floor(coeff))
            if _coeff != int(coeff):
                if assume_integer:
                    warnings.warn(f"Objective term {term} has non-integer coefficient. CPMpy will assume it is integer.")
                else:
                    raise ValueError(f"Objective term {term} has non-integer coefficient. CPMpy does not support non-integer coefficients.")
            obj_coeffs[i] = _coeff

        if direction == "minimize":
            model.minimize(cp.sum(obj_vars * obj_coeffs))
        elif direction == "maximize":
            model.maximize(cp.sum(obj_vars * obj_coeffs))
        else:
            raise ValueError(f"Unsupported objective sense: {direction}")

        return model



class _SCIPWriter:
    """
    A helper class aiding in translating CPMpy models to SCIP models.

    Borrows a lot of its implementation from the prototype SCIP solver interface from git branch `scip2`.

    TODO: code should be reused once SCIP has been added as a solver backend.
    """

    @staticmethod
    def supported():
        # try to import the package
        try:
            import pyscipopt as scip
            return True
        except:
            return False

    def __init__(self, problem_name: Optional[str] = None):
        if not self.supported():
            raise Exception(
                "SCIP: Install SCIP IO dependencies: cpmpy[io.scip]")
        import pyscipopt as scip

        self.scip_model = scip.Model(problem_name)

        self.user_vars = set() 
        self._varmap = dict()  # maps cpmpy variables to native solver variables
        self._csemap = dict()  # maps cpmpy expressions to solver expressions

        self._cons_counter = 0

    def solver_var(self, cpm_var):
        """
            Creates solver variable for cpmpy variable
            or returns from cache if previously created
        """
        if is_num(cpm_var): # shortcut, eases posting constraints
            return cpm_var

        # special case, negative-bool-view
        # work directly on var inside the view
        if isinstance(cpm_var, NegBoolView):
            raise Exception("Negative literals should not be part of any equation. See /transformations/linearize for more details")

        # create if it does not exit
        if cpm_var not in self._varmap:
            if isinstance(cpm_var, _BoolVarImpl):
                revar = self.scip_model.addVar(vtype='B', name=cpm_var.name)
            elif isinstance(cpm_var, _IntVarImpl):
                revar = self.scip_model.addVar(lb=cpm_var.lb, ub=cpm_var.ub, vtype='I', name=cpm_var.name)
            else:
                raise NotImplementedError("Not a known var {}".format(cpm_var))
            self._varmap[cpm_var] = revar

        # return from cache
        return self._varmap[cpm_var]


    def solver_vars(self, cpm_vars):
        """
           Like `solver_var()` but for arbitrary shaped lists/tensors
        """
        if is_any_list(cpm_vars):
            return [self.solver_vars(v) for v in cpm_vars]
        return self.solver_var(cpm_vars)

    def objective(self, expr, minimize=True):
        """
            Post the given expression to the solver as objective to minimize/maximize

            'objective()' can be called multiple times, only the last one is stored

            (technical side note: any constraints created during conversion of the objective
                are premanently posted to the solver)
        """

        # make objective function non-nested
        (flat_obj, flat_cons) = (flatten_objective(expr))
        self += flat_cons
        get_variables(flat_obj, collect=self.user_vars)  # add potentially created constraints

        # make objective function or variable and post
        obj = self._make_numexpr(flat_obj)
        if minimize:
            self.scip_model.setObjective(obj, sense='minimize')
        else:
            self.scip_model.setObjective(obj, sense='maximize')


    def _make_numexpr(self, cpm_expr):
        """
            Turns a numeric CPMpy 'flat' expression into a solver-specific
            numeric expression

            Used especially to post an expression as objective function
        """
        import pyscipopt as scip

        if is_num(cpm_expr):
            return cpm_expr

        # decision variables, check in varmap
        if isinstance(cpm_expr, _NumVarImpl):  # cp.boolvar is subclass of _NumVarImpl
            return self.solver_var(cpm_expr)

        # sum
        if cpm_expr.name == "sum":
            return scip.quicksum(self.solver_vars(cpm_expr.args))
        if cpm_expr.name == "sub":
            a,b = self.solver_vars(cpm_expr.args)
            return a - b
        # wsum
        if cpm_expr.name == "wsum":
            return scip.quicksum(w * self.solver_var(var) for w, var in zip(*cpm_expr.args))

        raise NotImplementedError("scip: Not a known supported numexpr {}".format(cpm_expr))
    
    
    def transform(self, cpm_expr):
        """
            Transform arbitrary CPMpy expressions to constraints the solver supports

            Implemented through chaining multiple solver-independent **transformation functions** from
            the `cpmpy/transformations/` directory.

            See the 'Adding a new solver' docs on readthedocs for more information.

        :param cpm_expr: CPMpy expression, or list thereof
        :type cpm_expr: Expression or list of Expression

        :return: list of Expression
        """
        # apply transformations, then post internally
        # expressions have to be linearized to fit in MIP model. See /transformations/linearize
        cpm_cons = toplevel_list(cpm_expr)
        supported = {"alldifferent"}  # alldiff has a specialized MIP decomp in linearize
        cpm_cons = decompose_in_tree(cpm_cons, supported)
        cpm_cons = flatten_constraint(cpm_cons)  # flat normal form
        cpm_cons = reify_rewrite(cpm_cons, supported=frozenset(['sum', 'wsum','sub']))  # constraints that support reification
        cpm_cons = only_numexpr_equality(cpm_cons, supported=frozenset(["sum", "wsum", "sub"]))  # supports >, <, !=
        cpm_cons = only_implies(cpm_cons)  # anything that can create full reif should go above...
        cpm_cons = linearize_constraint(cpm_cons, supported=frozenset({"sum", "wsum","sub", "mul", "div"})) # the core of the MIP-linearization
        cpm_cons = only_positive_bv(cpm_cons)  # after linearization, rewrite ~bv into 1-bv
        return cpm_cons

    def _get_constraint_name(self):
        name = f"cons_{self._cons_counter}"
        self._cons_counter += 1
        return name

    
    def add(self, cpm_expr_orig):
        """
                Eagerly add a constraint to the underlying solver.

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

        # add new user vars to the set
        get_variables(cpm_expr_orig, collect=self.user_vars)

        # transform and post the constraints
        for cpm_expr in self.transform(cpm_expr_orig):

            # Comparisons: only numeric ones as 'only_bv_implies()' has removed the '==' reification for Boolean expressions
            # numexpr `comp` bvar|const
            if isinstance(cpm_expr, Comparison):
                lhs, rhs = cpm_expr.args
                sciprhs = self.solver_var(rhs)

                # Thanks to `only_numexpr_equality()` only supported comparisons should remain
                if cpm_expr.name == '<=':
                    if (isinstance(lhs, Operator) and lhs.name == "sum" and all(a.is_bool() and not isinstance(a, NegBoolView) for a in lhs.args)):
                        if rhs == 1: # special SOS1 constraint?
                            self.scip_model.addConsSOS1(self.solver_vars(lhs.args), name=self._get_constraint_name())
                        else: # cardinality constraint
                            self.scip_model.addConsCardinality(self.solver_vars(lhs.args), rhs, name=self._get_constraint_name())
                    else:
                        sciplhs = self._make_numexpr(lhs)
                        self.scip_model.addCons(sciplhs <= sciprhs, name=self._get_constraint_name())

                elif cpm_expr.name == '>=':
                    sciplhs = self._make_numexpr(lhs)
                    self.scip_model.addCons(sciplhs >= sciprhs, name=self._get_constraint_name())
                elif cpm_expr.name == '==':
                    if isinstance(lhs, _NumVarImpl) \
                            or (isinstance(lhs, Operator) and (lhs.name == 'sum' or lhs.name == 'wsum' or lhs.name == "sub")):
                        # a BoundedLinearExpression LHS, special case, like in objective
                        sciplhs = self._make_numexpr(lhs)
                        self.scip_model.addCons(sciplhs == sciprhs, name=self._get_constraint_name())

                    elif lhs.name == 'mul':
                        scp_vars = self.solver_vars(lhs.args)
                        scp_lhs = scp_vars[0] * scp_vars[1]
                        for v in scp_vars[2:]:
                            scp_lhs *= v
                        self.scip_model.addCons(scp_lhs == sciprhs, name=self._get_constraint_name())

                    elif lhs.name == 'div':
                        a, b = self.solver_vars(lhs.args)
                        self.scip_model.addCons(a / b == sciprhs, name=self._get_constraint_name())

                    else:
                        raise NotImplementedError(
                            "Not a known supported scip comparison '{}' {}".format(lhs.name, cpm_expr))

                        # SCIP does have 'addConsAnd', 'addConsOr', 'addConsXor', 'addConsSOS2' #TODO?
                else:
                    raise NotImplementedError(
                    "Not a known supported scip comparison '{}' {}".format(lhs.name, cpm_expr))

            elif isinstance(cpm_expr, Operator) and cpm_expr.name == "->":
                # Indicator constraints
                # Takes form bvar -> sum(x,y,z) >= rvar
                cond, sub_expr = cpm_expr.args
                assert isinstance(cond, cp.boolvar), f"Implication constraint {cpm_expr} must have BoolVar as lhs"
                assert isinstance(sub_expr, Comparison), "Implication must have linear constraints on right hand side"

                lhs, rhs = sub_expr.args
                assert isinstance(lhs, _NumVarImpl) or lhs.name == "sum" or lhs.name == "wsum", f"Unknown linear expression {lhs} on right side of indicator constraint: {cpm_expr}"
                assert is_num(rhs), f"linearize should only leave constants on rhs of comparison but got {rhs}"

                if sub_expr.name == ">=":  # change sign
                    if lhs.name == "sum":
                        lhs = Operator("wsum", [[-1] * len(lhs.args), lhs.args])
                    elif lhs.name == "wsum":
                        lhs = Operator("wsum", [[-w for w in lhs.args[0]], lhs.args[1]])
                    else:
                        lhs = Operator("wsum",[[-1], [lhs]])
                    sub_expr = lhs <= -rhs

                if sub_expr.name == "<=":
                    lhs, rhs = sub_expr.args
                    lin_expr = self._make_numexpr(lhs)
                    if isinstance(cond, NegBoolView):
                        self.scip_model.addConsIndicator(lin_expr <= rhs, name=self._get_constraint_name(),
                                                        binvar=self.solver_var(cond._bv), activeone=False)
                    else:
                        self.scip_model.addConsIndicator(lin_expr <= rhs, name=self._get_constraint_name(),
                                                        binvar=self.solver_var(cond), activeone=True)

                elif sub_expr.name == "==": # split into <= and >=
                    # TODO: refactor to avoid re-transforming constraints?
                    self += [cond.implies(lhs <= rhs), cond.implies(lhs >= rhs)]
                else:
                    raise Exception(f"Unknown linear expression {sub_expr} name")

            # True or False
            elif isinstance(cpm_expr, BoolVal):
                # not sure how else to do it
                if cpm_expr.args[0] is False:
                    bv = self.solver_var(cp.boolvar())
                    self.scip_model.addCons(bv <= -1, name=self._get_constraint_name())

            # a direct constraint, pass to solver
            elif isinstance(cpm_expr, DirectConstraint):
                cpm_expr.callSolver(self, self.scip_model)

            else:
                raise NotImplementedError(cpm_expr)  # if you reach this... please report on github

        return self
    __add__ = add


def _to_writer(model: cp.Model, problem_name: Optional[str] = None) -> _SCIPWriter:
    """
    Convert a CPMpy model to a SCIP writer
    """
    writer = _SCIPWriter(problem_name=problem_name)
    # 1) post constraints
    for constraint in model.constraints:
        writer += constraint
    # 2) post objective
    if not model.has_objective():
        raise ValueError("Model has no objective function")
    writer.objective(model.objective_, model.objective_is_min)
    return writer


def to_scip(model: cp.Model) -> "pyscipopt.Model":
    """
    Convert a CPMpy model to a SCIP model

    Arguments:
        model: CPMpy model

    Returns:
        pyscipopt.Model: SCIP model
    """
    writer = _to_writer(model)
    return writer.scip_model


def _add_header(fname: os.PathLike, format: str, header: Optional[str] = None):
    """
    Add a header to a file.

    Arguments:
        fname: The path to the file to add the header to.
        format: The format of the file.
        header: The header to add.
    """

    with open(fname, "r") as f:
        lines = f.readlines()

    if format == "mps":
        header = ["* " + line + "\n" for line in header.splitlines()]
        lines = header + lines
        
    elif format == "lp":
        header = ["\\ " + line + "\n" for line in header.splitlines()]
        lines = header + lines

    elif format == "cip":
        header = ["# " + line + "\n" for line in header.splitlines()]
        lines = header + lines

    elif format == "fzn":
        header = ["% " + line + "\n" for line in header.splitlines()]
        lines = header + lines

    elif format == "gms":
        header = ["* " + line + "\n" for line in header.splitlines()]
        lines = [lines[0]] + header + lines[1:] # handle first line: $OFFLISTING

    elif format == "pip":
        header = ["\\ " + line + "\n" for line in header.splitlines()]
        lines = header + lines

    with open(fname, "w") as f:
        f.writelines(lines)


def write_scip(model: cp.Model, fname: Optional[str] = None, format: str = "mps", header: Optional[str] = None, verbose: bool = False) -> str:
    """
    Write a CPMpy model to file using a SCIP provided writer.
    Supported formats include: 
    - "mps"
    - "lp"
    - "cip"
    - "fzn"
    - "gms"
    - "pip"

    More formats can be supported upon the installation of additional dependencies (like SIMPL).
    For more information, see the SCIP documentation: https://pyscipopt.readthedocs.io/en/latest/tutorials/readwrite.html
    """

    writer = _to_writer(model, problem_name="CPMpy Model")
    
    # Decide where to write
    if fname is None:
        with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as tmp:
            fname = tmp.name
        try:
            writer.scip_model.writeProblem(fname)
            _add_header(fname, format, header)
            with open(fname, "r") as f:
                return f.read()
        finally:
            os.remove(fname)
    else:
        if not verbose: writer.scip_model.hideOutput()
        writer.scip_model.writeProblem(fname, verbose=verbose)
        if not verbose: writer.scip_model.hideOutput(quiet=False)
        _add_header(fname, format, header)
        with open(fname, "r") as f:
            return f.read()

def main():
    parser = argparse.ArgumentParser(description="Parse and solve a SCIP compatible model using CPMpy")
    parser.add_argument("model", help="Path to a SCIP compatible file (or raw string if --string is given)")
    parser.add_argument("-s", "--solver", default=None, help="Solver name to use (default: CPMpy's default)")
    parser.add_argument("--string", action="store_true", help="Interpret the first argument (model) as a raw OPB string instead of a file path")
    parser.add_argument("-t", "--time-limit", type=int, default=None, help="Time limit for the solver in seconds (default: no limit)")
    args = parser.parse_args()

    # Build the CPMpy model
    try:
        if args.string:
            model = read_scip(args.model)
        else:
            model = read_scip(os.path.expanduser(args.model))
    except Exception as e:
        sys.stderr.write(f"Error reading model: {e}\n")
        sys.exit(1)

    # Solve the model
    try:
        if args.solver:
            result = model.solve(solver=args.solver, time_limit=args.time_limit)
        else:
            result = model.solve(time_limit=args.time_limit)
    except Exception as e:
        sys.stderr.write(f"Error solving model: {e}\n")
        sys.exit(1)

    # Print results
    print("Status:", model.status())
    if result is not None:
        if model.has_objective():
            print("Objective:", model.objective_value())
    else:
        print("No solution found.")

if __name__ == "__main__":
    main()
