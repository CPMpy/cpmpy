#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## smtlib.py
##
"""
    This file implements helper functions for exporting CPMpy models to SMT-LIB format 
    and loading SMT-LIB models into CPMpy. Additionally, an "executor" class is provided 
    that can be used to execute solving-related commands (check-sat, get-value, get-model, etc.) 
    using a CPMpy solver backend.
    
    SMT-LIB is a standard textual format for SMT (Satisfiability Modulo Theories) problems.
    
    This tool uses pySMT to convert CPMpy models to and from SMT-LIB format.
    
    Example usage:
    
    1) Exporting CPMpy models to SMT-LIB:

    .. code-block:: python
    
        import cpmpy as cp
        from cpmpy.tools import smtlib
        
        x = cp.intvar(0, 10)
        y = cp.intvar(0, 10)
        m = cp.Model([x + y >= 5, x <= 3])
        
        # Export to SMT-LIB file
        smtlib.write_smtlib(m, "model.smt2")
        
        # Or get as string
        smtlib_str = smtlib.model_to_smtlib(m)
        print(smtlib_str)

    2) Reading SMT-LIB files into CPMpy:

    .. code-block:: python
    
        import cpmpy as cp
        from cpmpy.tools import smtlib
        
        model = smtlib.read_smtlib("model.smt2")
        model.solve()
        
        # Or read from string
        smtlib_content = '''
        (set-logic QF_LIA)
        (declare-fun x () Int)
        (declare-fun y () Int)
        (assert (>= (+ x y) 5))
        (assert (<= x 3))
        '''
        model = smtlib.smtlib_to_model(smtlib_content)
        model.solve()
        
    3) Using the interpreter for incremental model building:
    
    .. code-block:: python
    
        import cpmpy as cp
        from cpmpy.tools import smtlib
        
        interpreter = smtlib.SMTLibInterpreter()
        interpreter.interpret_string('(set-logic QF_LIA) (declare-fun x () Int) (assert (>= x 5)) (assert (<= x 10))')
        model = interpreter.get_model()
        model.solve()

    4) Using the executor to actually execute SMT-LIB commands:

    .. code-block:: python
    
        import cpmpy as cp
        from cpmpy.tools import smtlib
        
        executor = smtlib.SMTLibExecutor(solver_name="ortools")
        executor.interpret_string('''
            (set-logic QF_LIA)
            (declare-fun x () Int)
            (assert (>= x 5))
            (assert (<= x 10))
            (check-sat)
        ''')
        print("SAT result:", executor.last_sat_result)  # True or False
        model = executor.get_model()  # CPMpy model with solution values
        
        # You can use any CPMpy solver backend
        executor_z3 = smtlib.SMTLibExecutor(solver_name="z3")
        executor_gurobi = smtlib.SMTLibExecutor(solver_name="gurobi")
"""

import os
import warnings
from io import StringIO
from typing import Optional

import cpmpy as cp
from cpmpy.model import Model
from cpmpy.exceptions import NotSupportedError


def _pysmt_available():
    """
    Check if pySMT is available.
    (does not require any solver backends to be installed)
    """
    try:
        import pysmt
        return True
    except ModuleNotFoundError:
        return False


class _SMTLibConverter:
    """
    Converter from CPMpy to SMT-LIB format using CPM_pysmt solver backend.
    This converter uses the CPM_pysmt solver instance for transformations and conversions.
    """
    def __init__(self):
        # Create a CPM_pysmt solver instance to use for transformations and conversions
        # We use an empty model since we just need the transformation/conversion logic
        from cpmpy.solvers.pysmt import CPM_pysmt
        import cpmpy as cp
        
        # Create solver with empty model (we'll add constraints ourselves)
        self.solver = CPM_pysmt(cpm_model=cp.Model())
        
        # Note: We use the solver's _varmap directly for variable resolution
        # The solver maintains the mapping between CPMpy variables and pySMT symbols
    
    def _cpm_to_pysmt_expr(self, cpm_expr):
        """
        Convert a CPMpy expression to a pySMT formula using the solver's conversion logic.
        """
        # Use the solver's classmethod for conversion with our variable resolver
        from cpmpy.solvers.pysmt import CPM_pysmt
        return CPM_pysmt._cpm_to_pysmt_expr_impl(cpm_expr, var_resolver=self.solver.solver_var)
    
    def model_to_formula(self, model):
        """
        Convert a CPMpy model to a pySMT formula using the solver's transformation logic.
        """
        from pysmt.shortcuts import And, GE, LE, TRUE
        from cpmpy.expressions.variables import _IntVarImpl
        
        # Use solver's transform method to transform constraints
        constraint_formulas = []
        
        # Collect all variables (including from objective if present)
        all_vars = set()
        for constraint in model.constraints:
            from cpmpy.transformations.get_variables import get_variables
            get_variables(constraint, collect=all_vars)
        if model.has_objective():
            from cpmpy.transformations.get_variables import get_variables
            get_variables(model.objective_, collect=all_vars)
        
        # Create variable bounds constraints
        for cpm_var in all_vars:
            if isinstance(cpm_var, _IntVarImpl):
                # Use solver's solver_var to create/get variable (this adds bounds to solver, but that's ok)
                pysmt_var = self.solver.solver_var(cpm_var)
                # Create explicit bound constraints for the formula
                from pysmt.shortcuts import Int
                constraint_formulas.append(GE(pysmt_var, Int(cpm_var.lb)))
                constraint_formulas.append(LE(pysmt_var, Int(cpm_var.ub)))
        
        # Transform and convert model constraints using solver's methods
        for constraint in model.constraints:
            # Use solver's transform method (doesn't modify solver state, just returns transformed constraints)
            transformed = self.solver.transform(constraint)
            # Convert each transformed constraint
            for cpm_con in transformed:
                pysmt_expr = self.solver._pysmt_expr(cpm_con)
                constraint_formulas.append(pysmt_expr)
        
        if constraint_formulas: # create one big conjunction of all constraints
            return And(constraint_formulas) if len(constraint_formulas) > 1 else constraint_formulas[0]
        else:
            return TRUE() # empty model


def model_to_smtlib(model: Model, logic: Optional[str] = None, daggify: bool = True) -> str:
    """
    Converts a CPMpy model to SMT-LIB format string.
    
    This function uses pySMT's formula manipulation to convert the model to SMT-LIB format.
    It does NOT require any solver backends to be installed - only pySMT itself.
    
    Arguments:
        model: CPMpy Model to convert
        logic: Optional SMT-LIB logic name (e.g., "QF_LIA" for quantifier-free linear integer arithmetic).
               If None, pySMT will try to infer an appropriate logic.
        daggify: If True, uses DAG (directed acyclic graph) representation for shared subexpressions.
                 If False, uses tree representation. DAG is more compact but may be less readable.
    
    Returns:
        String containing the SMT-LIB representation of the model.
    
    Raises:
        ImportError: If pySMT is not installed.
        Exception: If the model cannot be converted to SMT-LIB format.
    
    Example:
    
    .. code-block:: python
    
        import cpmpy as cp
        from cpmpy.tools import smtlib
        
        x = cp.intvar(0, 10)
        y = cp.intvar(0, 10)
        m = cp.Model([x + y >= 5, x <= 3])
        
        smtlib_str = smtlib.model_to_smtlib(m)
        print(smtlib_str)
    """
    if not _pysmt_available():
        raise ImportError("pySMT is required for SMT-LIB export. Install it with: pip install pysmt")
    
    # Create standalone converter (no solver needed)
    converter = _SMTLibConverter()
    
    # Convert model to pySMT formula
    formula = converter.model_to_formula(model)
    
    # Get objective expression if present
    objective_expr = None
    minimize = True
    if model.has_objective():
        # Use solver's objective handling (with flattening, similar to solver.objective())
        from cpmpy.transformations.flatten_model import flatten_objective
        (flat_obj, flat_cons) = flatten_objective(model.objective_)
        # Add any constraints created during objective flattening to the formula
        # (Note: these are already handled in model_to_formula via model.constraints,
        # but if there are new constraints from flattening, we should include them)
        # Convert flattened objective
        objective_expr = converter.solver._pysmt_expr(flat_obj)
        minimize = model.objective_is_min
    
    # Convert formula to SMT-LIB script
    from pysmt.smtlib.script import smtlibscript_from_formula, SmtLibCommand
    from pysmt.smtlib import commands as smtcmd
    from pysmt.smtlib.annotations import Annotations
    
    script = smtlibscript_from_formula(formula, logic=logic)
    
    # Add annotations for variable names (use solver's varmap)
    annotations = Annotations()
    for cpm_var, pysmt_var in converter.solver._varmap.items():
        # Get the CPMpy variable name (use name attribute or str() as fallback)
        cpm_name = getattr(cpm_var, 'name', None) or str(cpm_var)
        # Add :named annotation with the CPMpy variable name
        annotations.add(pysmt_var, 'named', cpm_name)
    
    # Set annotations on the script
    script.annotations = annotations
    
    # Handle objective if present
    if objective_expr is not None:
        opt_type = smtcmd.MINIMIZE if minimize else smtcmd.MAXIMIZE
        opt_cmd = SmtLibCommand(opt_type, [objective_expr, []])
        
        # Insert objective command before check-sat if present
        check_sat_idx = None
        for i, cmd in enumerate(script.commands):
            if cmd.name == smtcmd.CHECK_SAT:
                check_sat_idx = i
                break
        
        if check_sat_idx is not None:
            script.commands.insert(check_sat_idx, opt_cmd)
        else:
            script.commands.append(opt_cmd)
    
    # Serialize to string
    buf = StringIO()
    script.serialize(buf, daggify=daggify)
    return buf.getvalue()


def write_smtlib(model: Model, fname: str, daggify: bool = True) -> None:
    """
    Writes a CPMpy model to an SMT-LIB format file.
    
    Arguments:
        model: CPMpy Model to convert
        fname: File path where the SMT-LIB file will be written
        daggify: If True, uses DAG (directed acyclic graph) representation for shared subexpressions.
                 If False, uses tree representation. DAG is more compact but may be less readable.
    
    Raises:
        ImportError: If pySMT is not installed.
        Exception: If the model cannot be converted to SMT-LIB format.
    
    Example:
    
    .. code-block:: python
    
        import cpmpy as cp
        from cpmpy.tools import smtlib
        
        x = cp.intvar(0, 10)
        y = cp.intvar(0, 10)
        m = cp.Model([x + y >= 5, x <= 3])
        
        smtlib.write_smtlib(m, "model.smt2")
    """
    smtlib_str = model_to_smtlib(model, daggify=daggify)
    
    with open(fname, 'w') as f:
        f.write(smtlib_str)


class _SMTLibReader:
    """
    Standalone converter from SMT-LIB format to CPMpy models.
    This converter does not require any solver backends - it only uses pySMT's formula manipulation.
    
    .. note::
        Not all SMT-LIB constraints can be converted to CPMpy. Unsupported constraints
        will raise `NotImplementedError` or be skipped with warnings.
    """
    def __init__(self, default_bounds=None):
        """
        Initialize the SMT-LIB reader.
        
        Arguments:
            default_bounds: Optional tuple (lower_bound, upper_bound) to use when bounds
                          are not detected from SMT-LIB assertions. Default is (-2**31, 2**31 - 1).
        """
        # Variable mapping: pySMT symbol -> CPMpy variable
        self._varmap = {}
        # Name mapping: symbol name -> CPMpy variable (for annotations)
        self._name_map = {}
        # Bound tracking: pySMT symbol -> (lower_bound, upper_bound) or None
        # Lower bound is from >= constraints, upper bound is from <= constraints
        self._var_bounds = {}
        # Default bounds to use when bounds are not detected
        if default_bounds is None:
            self._default_lb = -2**31
            self._default_ub = 2**31 - 1
            self._using_default_bounds = True  # Track if we're using default bounds
            # Warn once at initialization if using default bounds
            warnings.warn(f"Using default bounds [{self._default_lb}, {self._default_ub}] for variables without detected bounds from SMT-LIB assertions.")
        else:
            self._default_lb, self._default_ub = default_bounds
            self._using_default_bounds = False  # User provided custom bounds
        
    def _pysmt_to_cpm_expr(self, pysmt_expr):
        """
        Convert a pySMT expression to a CPMpy expression.
        """
        from pysmt.fnode import FNode
        from pysmt.typing import BOOL, INT
        import cpmpy as cp
        from cpmpy.expressions.variables import boolvar, intvar
        from cpmpy.expressions.core import Operator, Comparison, BoolVal
        from cpmpy.exceptions import NotSupportedError
        
        # Handle constants
        if pysmt_expr.is_constant():
            if pysmt_expr.is_bool_constant():
                return bool(pysmt_expr.constant_value())
            elif pysmt_expr.is_int_constant():
                return int(pysmt_expr.constant_value())
            else:
                raise NotSupportedError(f"Unsupported constant type: {pysmt_expr}")
        
        # Handle symbols (variables)
        elif pysmt_expr.is_symbol():
            symbol_name = pysmt_expr.symbol_name()

            # Get variable from cache (if already created)
            if pysmt_expr in self._varmap:
                return self._varmap[pysmt_expr]
            
            # Use the exact SMT-LIB symbol name (don't modify it)
            # This ensures the variable is posted to the solver correctly
            cpm_name = symbol_name
            
            # Create CPMpy variable based on type
            if pysmt_expr.symbol_type() == BOOL:
                cpm_var = boolvar(name=cpm_name)
            elif pysmt_expr.symbol_type() == INT:
                # Try to detect bounds from assertions if available
                lb, ub = self._get_inferred_bounds(pysmt_expr)
                if lb is not None or ub is not None:
                    # Use detected bounds, with reasonable defaults for missing ones
                    if lb is None:
                        lb = self._default_lb
                    if ub is None:
                        ub = self._default_ub
                    cpm_var = intvar(lb, ub, name=cpm_name)
                else:
                    # Use default bounds if bounds not detected
                    lb = self._default_lb
                    ub = self._default_ub
                    cpm_var = intvar(lb, ub, name=cpm_name)
            else:
                raise NotSupportedError(f"Unsupported variable type: {pysmt_expr.symbol_type()}")
            
            # Store mapping - this ensures we only create the variable once with the best bounds
            self._varmap[pysmt_expr] = cpm_var
            return cpm_var
        
        # Handle operators
        # - and
        elif pysmt_expr.is_and():
            args = [self._pysmt_to_cpm_expr(arg) for arg in pysmt_expr.args()]
            return cp.all(args)
        # - or
        elif pysmt_expr.is_or():
            args = [self._pysmt_to_cpm_expr(arg) for arg in pysmt_expr.args()]
            return cp.any(args)
        # - not
        elif pysmt_expr.is_not():
            arg = self._pysmt_to_cpm_expr(pysmt_expr.arg(0))
            return ~arg
        # - implies
        elif pysmt_expr.is_implies():
            left = self._pysmt_to_cpm_expr(pysmt_expr.arg(0))
            right = self._pysmt_to_cpm_expr(pysmt_expr.arg(1))
            return left.implies(right)
        # - ite
        elif pysmt_expr.is_ite():
            cond = self._pysmt_to_cpm_expr(pysmt_expr.arg(0))
            then_expr = self._pysmt_to_cpm_expr(pysmt_expr.arg(1))
            else_expr = self._pysmt_to_cpm_expr(pysmt_expr.arg(2))
            # Check if ite is boolean or numeric
            from cpmpy.expressions.utils import is_boolexpr
            if is_boolexpr(then_expr) and is_boolexpr(else_expr):
                from cpmpy.expressions.globalconstraints import IfThenElse
                return IfThenElse(cond, then_expr, else_expr)
            else:
                # Numeric ite - use decomposition with auxiliary variable
                # Store constraints for later addition to model
                from cpmpy.expressions.variables import intvar
                from cpmpy.expressions.utils import get_bounds
                lb_then, ub_then = get_bounds(then_expr)
                lb_else, ub_else = get_bounds(else_expr)
                aux_var = intvar(min(lb_then, lb_else), max(ub_then, ub_else))
                # Store decomposition constraints - these will be added when processing assertions
                if not hasattr(self, '_ite_constraints'):
                    self._ite_constraints = []
                self._ite_constraints.append(cond.implies(then_expr == aux_var))
                self._ite_constraints.append((~cond).implies(else_expr == aux_var))
                return aux_var
        
        # Handle arithmetic operations
        # - plus
        elif pysmt_expr.is_plus():
            args = [self._pysmt_to_cpm_expr(arg) for arg in pysmt_expr.args()]
            if len(args) == 0:
                return 0
            elif len(args) == 1:
                return args[0]
            else:
                return cp.sum(args)
        # - minus
        elif pysmt_expr.is_minus():
            left = self._pysmt_to_cpm_expr(pysmt_expr.arg(0))
            right = self._pysmt_to_cpm_expr(pysmt_expr.arg(1))
            return left - right
        # - mul
        elif pysmt_expr.is_times():
            args = [self._pysmt_to_cpm_expr(arg) for arg in pysmt_expr.args()]
            if len(args) == 0:
                return 1
            elif len(args) == 1:
                return args[0]
            else:
                result = args[0]
                for arg in args[1:]:
                    result = result * arg
                return result
        # - div
        elif pysmt_expr.is_div():
            left = self._pysmt_to_cpm_expr(pysmt_expr.arg(0))
            right = self._pysmt_to_cpm_expr(pysmt_expr.arg(1))
            return left // right  # Integer division
        # - mod
        # TODO: check this
        # Check for mod using node_type (pySMT doesn't have is_mod())
        elif hasattr(pysmt_expr, 'node_type') and pysmt_expr.node_type() == 35:  # BV_UREM or similar
            # This might be bitvector mod, which we don't support
            raise NotSupportedError(f"Bitvector modulo operations are not supported: {pysmt_expr}")
        # Note: Regular integer mod might be represented differently in pySMT
        
        # Handle comparisons
        # - le
        elif pysmt_expr.is_le():
            left = self._pysmt_to_cpm_expr(pysmt_expr.arg(0))
            right = self._pysmt_to_cpm_expr(pysmt_expr.arg(1))
            return left <= right
        # - lt
        elif pysmt_expr.is_lt():
            left = self._pysmt_to_cpm_expr(pysmt_expr.arg(0))
            right = self._pysmt_to_cpm_expr(pysmt_expr.arg(1))
            return left < right
        # Note: pySMT normalizes comparisons, so (>= x 5) becomes (5 <= x)
        # and (> x 5) becomes (5 < x).
        # - equals
        elif pysmt_expr.is_equals():
            left = self._pysmt_to_cpm_expr(pysmt_expr.arg(0))
            right = self._pysmt_to_cpm_expr(pysmt_expr.arg(1))
            return left == right
        # - distinct (alldifferent)
        elif pysmt_expr.is_distinct():
            args = [self._pysmt_to_cpm_expr(arg) for arg in pysmt_expr.args()]
            return cp.AllDifferent(args)
        # - unsupported
        else:
            raise NotSupportedError(f"Unsupported pySMT expression type: {pysmt_expr} (node type: {pysmt_expr.node_type()})")
    
    def _get_inferred_bounds(self, pysmt_symbol):
        """
        Get inferred bounds for a pySMT symbol from tracked bound constraints.
        
        Returns:
            (lower_bound, upper_bound) tuple, where each can be None if not found
        """
        if pysmt_symbol in self._var_bounds:
            return self._var_bounds[pysmt_symbol]
        return (None, None)
    
    def _extract_bounds_from_constraint(self, pysmt_formula):
        """
        Extract bound constraints from a pySMT formula and track them.
        
        This looks for simple patterns like:
        - (>= var constant) -> lower bound
        - (<= var constant) -> upper bound
        - (> var constant) -> lower bound + 1
        - (< var constant) -> upper bound - 1
        """
        from pysmt.typing import INT
        
        # Check if this is a simple comparison with a constant
        # Need to handle both argument orders: (var <= const) and (const <= var)
        if pysmt_formula.is_le():
            arg0 = pysmt_formula.arg(0)
            arg1 = pysmt_formula.arg(1)
            
            # Case 1: (var <= const) -> upper bound
            if arg0.is_symbol() and arg0.symbol_type() == INT and arg1.is_constant() and arg1.is_int_constant():
                symbol = arg0
                ub = int(arg1.constant_value())
                if symbol not in self._var_bounds:
                    self._var_bounds[symbol] = (None, None)
                current_lb, current_ub = self._var_bounds[symbol]
                if current_ub is None or ub < current_ub:
                    self._var_bounds[symbol] = (current_lb, ub)
            # Case 2: (const <= var) -> lower bound
            elif arg1.is_symbol() and arg1.symbol_type() == INT and arg0.is_constant() and arg0.is_int_constant():
                symbol = arg1
                lb = int(arg0.constant_value())
                if symbol not in self._var_bounds:
                    self._var_bounds[symbol] = (None, None)
                current_lb, current_ub = self._var_bounds[symbol]
                if current_lb is None or lb > current_lb:
                    self._var_bounds[symbol] = (lb, current_ub)
        
        elif pysmt_formula.is_lt():
            arg0 = pysmt_formula.arg(0)
            arg1 = pysmt_formula.arg(1)
            
            # Case 1: (var < const) -> upper bound
            if arg0.is_symbol() and arg0.symbol_type() == INT and arg1.is_constant() and arg1.is_int_constant():
                symbol = arg0
                ub = int(arg1.constant_value()) - 1  # < means strictly less
                if symbol not in self._var_bounds:
                    self._var_bounds[symbol] = (None, None)
                current_lb, current_ub = self._var_bounds[symbol]
                if current_ub is None or ub < current_ub:
                    self._var_bounds[symbol] = (current_lb, ub)
            # Case 2: (const < var) -> lower bound
            elif arg1.is_symbol() and arg1.symbol_type() == INT and arg0.is_constant() and arg0.is_int_constant():
                symbol = arg1
                lb = int(arg0.constant_value()) + 1  # < means strictly less
                if symbol not in self._var_bounds:
                    self._var_bounds[symbol] = (None, None)
                current_lb, current_ub = self._var_bounds[symbol]
                if current_lb is None or lb > current_lb:
                    self._var_bounds[symbol] = (lb, current_ub)
        
        # Handle equality constraints: (= var const) -> both lower and upper bound
        elif pysmt_formula.is_equals():
            arg0 = pysmt_formula.arg(0)
            arg1 = pysmt_formula.arg(1)
            
            # Case 1: (= var const) -> both bounds set to const
            if arg0.is_symbol() and arg0.symbol_type() == INT and arg1.is_constant() and arg1.is_int_constant():
                symbol = arg0
                value = int(arg1.constant_value())
                if symbol not in self._var_bounds:
                    self._var_bounds[symbol] = (None, None)
                # Set both bounds to the same value
                self._var_bounds[symbol] = (value, value)
            # Case 2: (= const var) -> both bounds set to const
            elif arg1.is_symbol() and arg1.symbol_type() == INT and arg0.is_constant() and arg0.is_int_constant():
                symbol = arg1
                value = int(arg0.constant_value())
                if symbol not in self._var_bounds:
                    self._var_bounds[symbol] = (None, None)
                # Set both bounds to the same value
                self._var_bounds[symbol] = (value, value)
        
        # Note: pySMT normalizes comparisons, so (>= x 5) becomes (5 <= x)
        # and (> x 5) becomes (5 < x).
        
        # Recursively check if this is an AND of constraints
        elif pysmt_formula.is_and():
            for arg in pysmt_formula.args():
                self._extract_bounds_from_constraint(arg)

        # Skip other formula types (they don't provide bound information)
        # Don't raise an error, just skip them
    
    def script_to_model(self, script):
        """
        Convert an SMT-LIB script to a CPMpy model.
        """
        # Extract annotations for variable names
        if script.annotations is not None:
            for formula, annots in script.annotations._annotations.items():
                if 'named' in annots and formula.is_symbol():
                    symbol_name = formula.symbol_name()
                    for name_value in annots['named']:
                        self._name_map[symbol_name] = str(name_value)
        
        # Use the interpreter to process commands, passing the default bounds
        default_bounds = (self._default_lb, self._default_ub) if hasattr(self, '_default_lb') else None
        interpreter = SMTLibInterpreter(default_bounds=default_bounds)
        # Share annotations
        interpreter._name_map = self._name_map
        
        for cmd in script.commands:
            interpreter.process_command(cmd)
        
        return interpreter.get_model()


class SMTLibInterpreter:
    """
    Interpreter for SMT-LIB commands that builds a CPMpy model incrementally.
    
    This class allows you to process SMT-LIB commands one by one, building up
    a CPMpy model incrementally. Unsupported commands will raise NotSupportedError.
    
    Example:
    
    .. code-block:: python
    
        from cpmpy.tools import smtlib
        
        interpreter = smtlib.SMTLibInterpreter()
        
        # Process commands from a script
        script = parser.get_script(StringIO(smtlib_content))
        for cmd in script.commands:
            interpreter.process_command(cmd)
        
        # Get the resulting model
        model = interpreter.get_model()
    """
    
    def __init__(self, extract_bounds_first=True, default_bounds=None):
        """
        Initialize the SMT-LIB interpreter.
        
        Arguments:
            extract_bounds_first: If True, extracts bounds from all assertions before creating variables
                                  (two-stage processing). If False, processes commands line-by-line.
                                  Default is True for better bound detection.
            default_bounds: Optional tuple (lower_bound, upper_bound) to use when bounds
                          are not detected from SMT-LIB assertions. Default is (-2**31, 2**31 - 1).
        """
        from pysmt.typing import BOOL, INT
        from cpmpy.expressions.variables import boolvar, intvar
        
        self.BOOL = BOOL
        self.INT = INT
        self.boolvar = boolvar
        self.intvar = intvar
        self.extract_bounds_first = extract_bounds_first
        
        # Variable mapping: pySMT symbol -> CPMpy variable
        self._varmap = {}
        # Name mapping: symbol name -> CPMpy variable (for annotations)
        self._name_map = {}
        # Constraints
        self._constraints = []
        # Objective
        self._objective = None
        self._minimize = True
        # Build model incrementally as commands are processed
        import cpmpy as cp
        self._model = cp.Model()
        # Converter for pySMT to CPMpy expressions
        self._reader = _SMTLibReader(default_bounds=default_bounds)
        # Share variable mapping with reader
        self._reader._varmap = self._varmap
        self._reader._name_map = self._name_map
        # Share bound tracking
        self._reader._var_bounds = {}
    
    def process_command(self, cmd):
        """
        Process a single SMT-LIB command.
        
        Arguments:
            cmd: SmtLibCommand object to process
        
        Raises:
            NotSupportedError: If the command is not supported
            Exception: If there's an error processing the command
        """
        from pysmt.smtlib import commands as smtcmd
        from cpmpy.exceptions import NotSupportedError
        
        if cmd.name == smtcmd.DECLARE_FUN or cmd.name == smtcmd.DECLARE_CONST:
            self._process_declare(cmd)
        elif cmd.name == smtcmd.ASSERT:
            self._process_assert(cmd)
        elif cmd.name == smtcmd.MINIMIZE:
            self._process_minimize(cmd)
        elif cmd.name == smtcmd.MAXIMIZE:
            self._process_maximize(cmd)
        elif cmd.name == smtcmd.SET_LOGIC:
            # Set logic command - we don't need to do anything with it
            pass
        elif cmd.name == smtcmd.SET_OPTION:
            # Set option command - we don't need to do anything with it
            pass
        elif cmd.name == smtcmd.SET_INFO:
            # Set info command - we don't need to do anything with it
            pass
        elif cmd.name == smtcmd.DEFINE_FUN:
            # Define function - not supported yet
            raise NotSupportedError(f"define-fun command is not supported: {cmd}")
        elif cmd.name == smtcmd.DEFINE_SORT:
            # Define sort - not supported yet
            raise NotSupportedError(f"define-sort command is not supported: {cmd}")
        elif cmd.name == smtcmd.DECLARE_SORT:
            # Declare sort - not supported yet
            raise NotSupportedError(f"declare-sort command is not supported: {cmd}")
        # Skip solving-related commands (will be handled by CPMpy's solver)
        elif cmd.name in [smtcmd.CHECK_SAT, smtcmd.CHECK_SAT_ASSUMING, smtcmd.CHECK_ALLSAT,
                         smtcmd.GET_VALUE, smtcmd.GET_MODEL, smtcmd.GET_ASSIGNMENT,
                         smtcmd.GET_ASSERTIONS, smtcmd.GET_UNSAT_CORE, smtcmd.GET_UNSAT_ASSUMPTIONS,
                         smtcmd.GET_PROOF, smtcmd.GET_OBJECTIVES, smtcmd.PUSH, smtcmd.POP,
                         smtcmd.RESET, smtcmd.RESET_ASSERTIONS, smtcmd.EXIT, smtcmd.ECHO]:
            # These are solving-related commands that will be handled by CPMpy's solver
            # Look at the executor class for more details on how to handle these commands.
            pass
        else:
            raise NotSupportedError(f"Unsupported SMT-LIB command: {cmd.name}")
    
    def _process_declare(self, cmd):
        """
        Process a declare-fun or declare-const command.
        """
        # Don't create variables immediately - let them be created lazily when first used
        # This allows bounds to be detected from assertions before variable creation
        # Variables will be created in _pysmt_to_cpm_expr when first encountered
        pass
    
    def _process_assert(self, cmd):
        """
        Process an assert command.
        """
        # Reset ITE constraints for this assertion : TODO: why?
        self._reader._ite_constraints = []
        
        pysmt_formula = cmd.args[0]
        
        # Extract bounds from constraint BEFORE converting
        # This ensures variables are created with the best available bounds
        self._reader._extract_bounds_from_constraint(pysmt_formula)
        
        # Now convert the constraint (variables will be created with best bounds)
        cpm_constraint = self._reader._pysmt_to_cpm_expr(pysmt_formula)
        self._constraints.append(cpm_constraint)
        
        # Add any ITE decomposition constraints
        self._constraints.extend(self._reader._ite_constraints)
        
        # Update model immediately
        self._model += cpm_constraint
        if self._reader._ite_constraints:
            self._model += self._reader._ite_constraints
    
    def _process_minimize(self, cmd):
        """
        Process a minimize command.
        """
        # Reset ITE constraints for this objective
        self._reader._ite_constraints = []
        
        self._objective = self._reader._pysmt_to_cpm_expr(cmd.args[0])
        self._minimize = True
        
        # Add any ITE decomposition constraints
        self._constraints.extend(self._reader._ite_constraints)
        
        # Update model immediately
        import cpmpy as cp
        self._model = cp.Model(self._model.constraints, minimize=self._objective)
        if self._reader._ite_constraints:
            self._model += self._reader._ite_constraints
    
    def _process_maximize(self, cmd):
        """
        Process a maximize command.
        """
        # Reset ITE constraints for this objective
        self._reader._ite_constraints = []
        
        self._objective = self._reader._pysmt_to_cpm_expr(cmd.args[0])
        self._minimize = False
        
        # Add any ITE decomposition constraints
        self._constraints.extend(self._reader._ite_constraints)
        
        # Update model immediately
        import cpmpy as cp
        self._model = cp.Model(self._model.constraints, maximize=self._objective)
        if self._reader._ite_constraints:
            self._model += self._reader._ite_constraints
    
    def get_model(self) -> cp.Model:
        """
        Get the CPMpy model built from the processed commands.
        
        Returns:
            A CPMpy Model object (built incrementally as commands are processed)
        """
        return self._model
    
    def reset(self):
        """
        Reset the interpreter to its initial state.
        """
        import cpmpy as cp
        
        self._varmap.clear()
        self._name_map.clear()
        self._constraints.clear()
        self._objective = None
        self._minimize = True
        self._model = cp.Model()
    
    def interpret(self, smt: str, open=open, extract_bounds_first=None) -> cp.Model:
        """
        Interpret an entire SMT-LIB instance.
        
        This method parses the SMT-LIB instance and processes all commands,
        building up a CPMpy model incrementally.
        
        Arguments:
            smt (str or os.PathLike):
                - A file path to an SMT-LIB file
                - OR a string containing the SMT-LIB content directly
            open: (callable):
                If `smt` is the path to a file, a callable to "open" that file (default=python standard library's 'open').
            extract_bounds_first: If True, extracts bounds from all assertions before creating variables
                                  (two-stage processing). If False, processes commands line-by-line.
                                  If None, uses the value from __init__. Default is None.

        Returns:
            A CPMpy model representing the SMT-LIB instance.
        
        Raises:
            ImportError: If pySMT is not installed.
            FileNotFoundError: If the instance does not exist.
            NotSupportedError: If any command is not supported.
            Exception: If there's an error processing the instance.
        """
        if not _pysmt_available():
            raise ImportError("pySMT is required for SMT-LIB interpretation. Install it with: pip install pysmt")
        

        # If smt is a path to a file -> open file
        if isinstance(smt, (str, os.PathLike)) and os.path.exists(smt):
            if open is not None:
                f = open(smt)
            else:
                f = _std_open(smt, "rt")
        # If smt is a string containing a model -> create a memory-mapped file
        else:
            f = StringIO(smt)

        from pysmt.smtlib.parser import SmtLibParser
        
        # Parse SMT-LIB instance from file or string into a script
        parser = SmtLibParser()
        script = parser.get_script_fname(f)
        
        # Extract annotations for variable names
        if script.annotations is not None:
            for formula, annots in script.annotations._annotations.items():
                if 'named' in annots and formula.is_symbol():
                    symbol_name = formula.symbol_name()
                    for name_value in annots['named']:
                        self._name_map[symbol_name] = str(name_value)
        
        # Use instance default if not specified
        if extract_bounds_first is None:
            extract_bounds_first = self.extract_bounds_first
        
        # First pass: extract bounds from all assertions before creating variables (if enabled)
        # This ensures variables are created with the tightest possible bounds
        if extract_bounds_first:
            from pysmt.smtlib import commands as smtcmd
            for cmd in script.commands:
                if cmd.name == smtcmd.ASSERT:
                    # Extract bounds without creating variables yet
                    self._reader._extract_bounds_from_constraint(cmd.args[0])
        
        # Process all commands (variables will be created with best bounds if extract_bounds_first=True)
        for cmd in script.commands:
            self.process_command(cmd)
        
        return self.get_model()
        
    def interpret_script(self, script, extract_bounds_first=None) -> cp.Model:
        """
        Interpret an SMT-LIB script object.
        
        This method processes all commands from a parsed SMT-LIB script,
        building up a CPMpy model incrementally.
        
        Arguments:
            script: SmtLibScript object to interpret
            extract_bounds_first: If True, extracts bounds from all assertions before creating variables
                                  (two-stage processing). If False, processes commands line-by-line.
                                  If None, uses the value from __init__. Default is None.
    
        Returns:
            A CPMpy model representing the SMT-LIB script.
        
        Raises:
            NotSupportedError: If any command is not supported.
            Exception: If there's an error processing the script.
        """
        # Extract annotations for variable names
        if script.annotations is not None:
            for formula, annots in script.annotations._annotations.items():
                if 'named' in annots and formula.is_symbol():
                    symbol_name = formula.symbol_name()
                    for name_value in annots['named']:
                        self._name_map[symbol_name] = str(name_value)
        
        # Use instance default if not specified
        if extract_bounds_first is None:
            extract_bounds_first = self.extract_bounds_first
        
        # First pass: extract bounds from all assertions before creating variables (if enabled)
        # This ensures variables are created with the tightest possible bounds
        if extract_bounds_first:
            from pysmt.smtlib import commands as smtcmd
            for cmd in script.commands:
                if cmd.name == smtcmd.ASSERT:
                    # Extract bounds without creating variables yet
                    self._reader._extract_bounds_from_constraint(cmd.args[0])
        
        # Process all commands (variables will be created with best bounds if extract_bounds_first=True)
        for cmd in script.commands:
            self.process_command(cmd)
        
        return self.get_model()


class SMTLibExecutor(SMTLibInterpreter):
    """
    Executor for SMT-LIB commands that builds a CPMpy model and executes solving commands.
    
    This class extends SMTLibInterpreter to actually execute solving-related commands
    (check-sat, get-value, get-model, etc.) using a CPMpy solver backend.
    
    Example:
    
    .. code-block:: python
    
        from cpmpy.tools import smtlib
        
        executor = smtlib.SMTLibExecutor(solver_name="ortools")
        executor.interpret_string('''
            (set-logic QF_LIA)
            (declare-fun x () Int)
            (assert (>= x 5))
            (check-sat)
            (get-value (x))
        ''')
        
        # Get the result of the last check-sat
        sat_result = executor.last_sat_result  # True, False, or None
        # Get the CPMpy model
        model = executor.get_model()
    """
    
    def __init__(self, solver_name=None, extract_bounds_first=True, default_bounds=None, **solver_options):
        """
        Initialize the SMT-LIB executor.
        
        Arguments:
            solver_name: Name of the CPMpy solver to use (e.g., "ortools", "z3", "gurobi").
                        If None, uses CPMpy's default solver.
            extract_bounds_first: If True, extracts bounds from all assertions before creating variables
                                  (two-stage processing). If False, processes commands line-by-line.
                                  Default is True for better bound detection.
            default_bounds: Optional tuple (lower_bound, upper_bound) to use when bounds
                          are not detected from SMT-LIB assertions. Default is (-2**31, 2**31 - 1).
            **solver_options: Additional options to pass to the CPMpy solver constructor.
        """
        super().__init__(extract_bounds_first=extract_bounds_first, default_bounds=default_bounds)
        
        import cpmpy as cp
        from cpmpy.solvers.utils import SolverLookup
        
        # Create CPMpy solver instance (will be initialized with model when first solve is called)
        self.solver_name = solver_name
        self.solver_options = solver_options
        self.cpm_solver = None  # Will be created lazily when needed
        
        # Track last solving result
        self.last_sat_result = None
        # Track last model with values
        self.last_model = None
        # Track results from get-value commands
        self.last_values = {}
        # Track execution results for commands
        self.command_results = []
        # Track push/pop levels for incremental solving
        self._push_levels = 0
    
    def _get_or_create_solver(self):
        """
        Get or create the CPMpy solver instance.
        """
        import cpmpy as cp
        from cpmpy.solvers.utils import SolverLookup
        
        if self.cpm_solver is None:
            # Get the current CPMpy model
            model = self.get_model()
            # Create solver instance
            self.cpm_solver = SolverLookup.get(self.solver_name, model, **self.solver_options)
        return self.cpm_solver
    
    def process_command(self, cmd):
        """
        Process a single SMT-LIB command, executing solving commands when encountered.
        
        Arguments:
            cmd: SmtLibCommand object to process
        
        Returns:
            Result of executing the command (if applicable), or None
        
        Raises:
            NotSupportedError: If the command is not supported
            Exception: If there's an error processing or executing the command
        """
        from pysmt.smtlib import commands as smtcmd
        from cpmpy.exceptions import NotSupportedError
        
        # First, try to process as a regular interpreter command (assertions, declarations, etc.)
        # If it's a solving command, we'll handle it below
        if cmd.name not in [smtcmd.CHECK_SAT, smtcmd.CHECK_SAT_ASSUMING, smtcmd.CHECK_ALLSAT,
                           smtcmd.GET_VALUE, smtcmd.GET_MODEL, smtcmd.GET_ASSIGNMENT,
                           smtcmd.GET_ASSERTIONS, smtcmd.GET_UNSAT_CORE, smtcmd.GET_UNSAT_ASSUMPTIONS,
                           smtcmd.GET_PROOF, smtcmd.GET_OBJECTIVES, smtcmd.PUSH, smtcmd.POP,
                           smtcmd.RESET, smtcmd.RESET_ASSERTIONS, smtcmd.EXIT, smtcmd.ECHO]:
            # Process normally (will raise NotSupportedError if not supported)
            result = super().process_command(cmd)
            # If a constraint was added, invalidate the solver (need to recreate with new model)
            if cmd.name == smtcmd.ASSERT or cmd.name == smtcmd.MINIMIZE or cmd.name == smtcmd.MAXIMIZE:
                self.cpm_solver = None  # Force recreation on next solve
            return result
        
        # Handle solving-related commands
        result = None
        
        if cmd.name == smtcmd.CHECK_SAT:
            # Execute check-sat using CPMpy solver
            try:
                solver = self._get_or_create_solver()
                self.last_sat_result = solver.solve()
                result = "sat" if self.last_sat_result else "unsat"
                # Store the model with values
                if self.last_sat_result:
                    self.last_model = self.get_model()
            except Exception as e:
                # Some solvers may raise exceptions for unsat or timeout
                self.last_sat_result = False
                result = "unsat"
            self.command_results.append(("check-sat", result))
        
        elif cmd.name == smtcmd.CHECK_SAT_ASSUMING:
            # Execute check-sat with assumptions
            assumptions = cmd.args
            try:
                solver = self._get_or_create_solver()
                # Note: Not all CPMpy solvers support assumptions
                self.last_sat_result = solver.solve(assumptions=assumptions)
                result = "sat" if self.last_sat_result else "unsat"
                if self.last_sat_result:
                    self.last_model = self.get_model()
            except Exception as e:
                self.last_sat_result = False
                result = "unsat"
            self.command_results.append(("check-sat-assuming", result))
        
        elif cmd.name == smtcmd.GET_VALUE:
            # Get values for specified terms (pySMT symbols need to be converted to CPMpy variables)
            terms = cmd.args
            if self.last_sat_result is not True:
                raise RuntimeError("Cannot get values: last check-sat was not sat")
            if self.last_model is None:
                raise RuntimeError("No model available")
            
            values = {}
            for term in terms:
                # Convert pySMT symbol to CPMpy variable via varmap
                if term in self._reader._varmap:
                    cpm_var = self._reader._varmap[term]
                    # Get value from the variable (solver sets _value attribute after solve)
                    try:
                        var_value = getattr(cpm_var, '_value', None)
                        if var_value is not None:
                            values[term] = var_value
                        else:
                            raise RuntimeError(f"Variable {term} has no value")
                    except AttributeError:
                        raise RuntimeError(f"Variable {term} has no value")
                else:
                    raise RuntimeError(f"Term {term} not found in variable map")
            self.last_values = values
            result = values
            self.command_results.append(("get-value", values))
        
        elif cmd.name == smtcmd.GET_MODEL:
            # Get full model (CPMpy model with values)
            if self.last_sat_result is not True:
                raise RuntimeError("Cannot get model: last check-sat was not sat")
            if self.last_model is None:
                self.last_model = self.get_model()
            result = self.last_model
            self.command_results.append(("get-model", result))
        
        elif cmd.name == smtcmd.GET_ASSERTIONS:
            # Get all assertions (return CPMpy constraints)
            result = list(self._constraints)
            self.command_results.append(("get-assertions", result))
        
        elif cmd.name == smtcmd.GET_UNSAT_CORE:
            # Get unsat core
            if self.last_sat_result is not False:
                raise RuntimeError("Cannot get unsat core: last check-sat was not unsat")
            try:
                solver = self._get_or_create_solver()
                if hasattr(solver, 'get_core'):
                    core = solver.get_core()
                    result = core
                else:
                    raise NotSupportedError("Solver does not support unsat core extraction")
            except Exception as e:
                raise RuntimeError(f"Error getting unsat core: {e}")
            self.command_results.append(("get-unsat-core", result))
        
        # elif cmd.name == smtcmd.PUSH:
        #     # Push solver state (if solver supports incremental solving)
        #     levels = cmd.args[0] if cmd.args else 1
        #     try:
        #         solver = self._get_or_create_solver()
        #         if hasattr(solver, 'push'):
        #             solver.push(levels)
        #             self._push_levels += levels
        #             result = None
        #         else:
        #             raise NotSupportedError("Solver does not support push/pop")
        #     except Exception as e:
        #         raise RuntimeError(f"Error pushing solver state: {e}")
        #     self.command_results.append(("push", levels))
        
        # elif cmd.name == smtcmd.POP:
        #     # Pop solver state
        #     levels = cmd.args[0] if cmd.args else 1
        #     try:
        #         solver = self._get_or_create_solver()
        #         if hasattr(solver, 'pop'):
        #             solver.pop(levels)
        #             self._push_levels = max(0, self._push_levels - levels)
        #             result = None
        #         else:
        #             raise NotSupportedError("Solver does not support push/pop")
        #     except Exception as e:
        #         raise RuntimeError(f"Error popping solver state: {e}")
        #     self.command_results.append(("pop", levels))
        
        elif cmd.name == smtcmd.RESET:
            # Reset solver
            try:
                # Reset interpreter state
                self.reset()
                # Recreate solver
                self.cpm_solver = None
                self._push_levels = 0
                result = None
            except Exception as e:
                raise RuntimeError(f"Error resetting solver: {e}")
            self.command_results.append(("reset", None))
        
        elif cmd.name == smtcmd.RESET_ASSERTIONS:
            # Reset assertions
            try:
                # Clear constraints
                self._constraints.clear()
                # Recreate solver
                self.cpm_solver = None
                result = None
            except Exception as e:
                raise RuntimeError(f"Error resetting assertions: {e}")
            self.command_results.append(("reset-assertions", None))
        
        elif cmd.name == smtcmd.EXIT:
            # Exit solver (cleanup)
            try:
                if self.cpm_solver is not None:
                    # Some solvers have exit/cleanup methods
                    if hasattr(self.cpm_solver, 'exit'):
                        self.cpm_solver.exit()
                    self.cpm_solver = None
                result = None
            except Exception as e:
                raise RuntimeError(f"Error exiting solver: {e}")
            self.command_results.append(("exit", None))
        
        elif cmd.name == smtcmd.ECHO:
            # Echo command - just return the string
            result = cmd.args[0] if cmd.args else ""
            self.command_results.append(("echo", result))
        
        else:
            # These commands are not yet supported
            raise NotSupportedError(f"SMT-LIB command '{cmd.name}' is not yet supported in executor mode")
        
        return result
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup solver."""
        try:
            if self.cpm_solver is not None:
                if hasattr(self.cpm_solver, 'exit'):
                    self.cpm_solver.exit()
                self.cpm_solver = None
        except:
            pass
        return False

_std_open = open
def read_smtlib(smt: str, open=open, default_bounds=None) -> cp.Model:
    """
    Reads an SMT-LIB instance and converts it to a CPMpy model.
    
    This function uses pySMT's parser to read SMT-LIB files and converts them to CPMpy models.
    It does NOT require any solver backends to be installed - only pySMT itself.
    
    .. note::
        Not all SMT-LIB constraints can be converted to CPMpy. Unsupported constraints
        will raise a `NotSupportedError` exception. Solving-related commands (check-sat, get-value, etc.)
        are silently skipped. 
    
    The function supports:
    - Integer and Boolean variables
    - Basic arithmetic operations (+, -, *, /, mod)
    - Comparisons (<=, <, >=, >, ==, !=)
    - Boolean operations (and, or, not, implies, ite)
    - AllDifferent (distinct)
    - Minimize/Maximize objectives
    
    Arguments:
        smt (str or os.PathLike):
            - A file path to an SMT-LIB file
            - OR a string containing the SMT-LIB content directly
        open: (callable):
            If `smt` is the path to a file, a callable to "open" that file (default=python standard library's 'open').
        default_bounds (tuple): 
            Optional tuple (lower_bound, upper_bound) to use when bounds
            are not detected from SMT-LIB assertions. Default is (-2**31, 2**31 - 1).
    
    Returns:
        A CPMpy Model object representing the SMT-LIB problem instance
    
    Raises:
        ImportError: If pySMT is not installed.
        FileNotFoundError: If the file does not exist.
        Exception: If the file cannot be parsed or converted.
    
    Example:
    
    .. code-block:: python
    
        import cpmpy as cp
        from cpmpy.tools import smtlib
        
        # Read SMT-LIB file
        model = smtlib.read_smtlib("model.smt2")
        
        # Solve the model
        model.solve()
    """
    if not _pysmt_available():
        raise ImportError("pySMT is required for SMT-LIB import. Install it with: pip install pysmt")

    from pysmt.smtlib.parser import SmtLibParser
    
    # Parse SMT-LIB file
    parser = SmtLibParser()
    
    # If smt is a path to a file -> use filename directly
    if isinstance(smt, (str, os.PathLike)) and os.path.exists(smt):
        script = parser.get_script_fname(smt)
    # If smt is a string containing a model -> create a StringIO and use get_script
    else:
        f = StringIO(smt)
        script = parser.get_script(f)
    
    # Convert script to CPMpy model
    reader = _SMTLibReader(default_bounds=default_bounds)
    return reader.script_to_model(script)

def execute_smtlib(smt: str, open=open, default_bounds=None) -> cp.Model:
    """
    Executes an SMT-LIB instance and returns a CPMpy model.
    
    This function uses the SMTLibExecutor to execute an SMT-LIB instance and returns a CPMpy model.
    """
    if not _pysmt_available():
        raise ImportError("pySMT is required for SMT-LIB import. Install it with: pip install pysmt")

    # If opb is a path to a file -> open file
    if isinstance(smt, (str, os.PathLike)) and os.path.exists(smt):
        if open is not None:
            f = open(smt)
        else:
            f = _std_open(smt, "rt")
    # If smt is a string containing a model -> create a memory-mapped file
    else:
        f = StringIO(smt)

    executor = SMTLibExecutor(default_bounds=default_bounds)
    return executor.interpret(smt, open=open, default_bounds=default_bounds)


