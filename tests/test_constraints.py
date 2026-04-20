import inspect

import cpmpy as cp
import numpy as np
from cpmpy import Model, SolverLookup, BoolVal
from cpmpy.transformations.get_variables import get_variables
from cpmpy.expressions.utils import argval, is_num, eval_comparison, get_bounds
from cpmpy.expressions.core import Comparison, Operator
from cpmpy.expressions.globalconstraints import GlobalConstraint
from cpmpy.expressions.globalfunctions import GlobalFunction
from cpmpy.expressions.variables import _IntVarImpl, _BoolVarImpl

import pytest


@pytest.fixture(autouse=True)
def reset_var_counters():
    """Reset the intvar and boolvar counters before each test."""
    _IntVarImpl.counter = 0
    _BoolVarImpl.counter = 0

from utils import skip_on_missing_pblib

# CHANGE THIS if you want test a different solver
#   make sure that `SolverLookup.get(solver)` works
# also add exclusions to the 3 EXCLUDE_* below as needed
SOLVERNAMES = [name for name, solver in SolverLookup.base_solvers() if solver.supported()]
ALL_SOLS = False # test whether all solutions returned by the solver satisfy the constraint
# ALL_SOLS = True # test whether all solutions returned by the solver satisfy the constraint

# Exclude some global constraints for solvers
NUM_GLOBAL = {
    "AllEqual", "AllDifferent", "AllDifferentExcept0",
    "AllDifferentExceptN", "AllEqualExceptN",
    "GlobalCardinalityCount", "InDomain", "Inverse", "Circuit",
    "Table", 'NegativeTable', "ShortTable", "Regular",
    "Increasing", "IncreasingStrict", "Decreasing", "DecreasingStrict", 
    "Precedence", "Cumulative", "NoOverlap", "CumulativeOptional", "NoOverlapOptional",
    "LexLess", "LexLessEq", "LexChainLess", "LexChainLessEq",
    # also global functions
    "Abs", "Element", "Minimum", "Maximum", "Count", "Among", "NValue", "NValueExcept", "Division", "Modulo", "Power"
}

# Solvers not supporting arithmetic constraints (numeric comparisons)
SAT_SOLVERS = {"pysdd"}

EXCLUDE_GLOBAL = {
                  "pysdd": NUM_GLOBAL | {"Xor"},
                  "minizinc": {"IncreasingStrict"}, # bug #813 reported on libminizinc
                  }
# EXCLUDE_GLOBAL = True

# Exclude certain operators for solvers.
# Not all solvers support all operators in CPMpy
EXCLUDE_OPERATORS = {"pysdd": {"sum", "wsum", "sub", "abs", "mul","-"},
                     }

# Variables to use in the rest of the test script
NUM_ARGS = [cp.intvar(-3, 5, name=n) for n in "xyz"]   # Numerical variables
SMALL_NUM_ARG = [cp.intvar(-2, 2, name=n) for n in "w"]   # Small domain numerical vars
NN_VAR = cp.intvar(0, 10, name="n_neg")                # Non-negative variable, needed in power functions
POS_VAR = cp.intvar(1,10, name="s_pos")                # A strictly positive variable
NUM_VAR = cp.intvar(0, 10, name="l")                   # A numerical variable

BOOL_ARGS = [cp.boolvar(name=n) for n in "abc"]        # Boolean variables
BOOL_VAR = cp.boolvar(name="p")                        # A boolean variable


def numexprs(solver):
    """
    Generate all numerical expressions
    Numexpr:
        - Operator (non-Boolean) with all args Var/constant (examples: +,*,/,mod,wsum)
                                                           (CPMpy class 'Operator', not is_bool())
        - Global functions  (examples: Max,Min,Element)    (CPMpy class 'GlobalFunction')
    """
    names = [(name, arity) for name, (arity, is_bool) in Operator.allowed.items() if not is_bool]
    if solver in EXCLUDE_OPERATORS:
        names = [(name, arity) for name, arity in names if name not in EXCLUDE_OPERATORS[solver]]
    for name, arity in names:
        if name == "wsum":
            yield Operator("wsum", [list(range(len(NUM_ARGS))), NUM_ARGS])
            yield Operator("wsum", [[True, BoolVal(False), np.True_], NUM_ARGS]) # bit of everything
            continue
        elif arity != 0:
            yield Operator(name, NUM_ARGS[:arity])
        else:
            yield Operator(name, NUM_ARGS)

    # boolexprs are also numeric
    for expr in bool_exprs(solver):
        yield expr
    
    if EXCLUDE_GLOBAL is not True:
        for expr in global_functions(solver):
            yield expr


# Generate all possible comparison constraints
def comp_constraints(solver):
    """
        Generate all comparison constraints
        - Numeric equality:  Numexpr == Var                (CPMpy class 'Comparison')
                         Numexpr == Constant               (CPMpy class 'Comparison')
        - Numeric disequality: Numexpr != Var              (CPMpy class 'Comparison')
                           Numexpr != Constant             (CPMpy class 'Comparison')
        - Numeric inequality (>=,>,<,<=): Numexpr >=< Var  (CPMpy class 'Comparison')
                                          Var >=< NumExpr  (CPMpy class 'Comparison')
    """
    for comp_name in sorted(Comparison.allowed):

        for numexpr in numexprs(solver):
            # numeric vs bool/num var/val (incl global func)
            for rhs in [NUM_VAR, BOOL_VAR, BoolVal(True), 1]:
                if solver in SAT_SOLVERS and not is_num(rhs):
                    continue
                for x,y in [(numexpr,rhs), (rhs,numexpr)]:
                    # check if the constraint we are trying to construct is always UNSAT
                    if any(eval_comparison(comp_name, xb,yb) for xb in get_bounds(x) for yb in get_bounds(y)):
                        yield Comparison(comp_name, x, y)
                    else: # impossible comparison, skip
                        pass

# Generate all possible boolean expressions
def bool_exprs(solver):
    """
        Generate all boolean expressions:
        - Boolean operators: and([Var]), or([Var])              (CPMpy class 'Operator', is_bool())
        - Boolean equality: Var == Var                          (CPMpy class 'Comparison')
        - Global constraints
    """

    names = [(name, arity) for name, (arity, is_bool) in Operator.allowed.items() if is_bool]
    if solver in EXCLUDE_OPERATORS:
        names = [(name, arity) for name, arity in names if name not in EXCLUDE_OPERATORS[solver]]

    for name, arity in names:
        if arity != 0:
            operator_args = BOOL_ARGS[:arity]
        else:
            operator_args = BOOL_ARGS

        yield Operator(name, operator_args)
        # Negated boolean values
        yield Operator(name, [~ arg for arg in operator_args])

    for eq_name in ["==", "!="]:
        yield Comparison(eq_name, *BOOL_ARGS[:2])

    if EXCLUDE_GLOBAL is not True:
        for cpm_cons in global_constraints(solver):
            yield cpm_cons

def global_constraints(solver):
    """
        Generate all global constraints
        -  AllDifferent, AllEqual, Circuit,  Minimum, Maximum, Element,
           Xor, Cumulative, NValue, Count, Increasing, Decreasing, IncreasingStrict, DecreasingStrict, LexLessEq, LexLess
    """
    classes = inspect.getmembers(cp.expressions.globalconstraints, inspect.isclass)
    # classes = [("Xor",cp.Xor)]
    classes = [(name, cls) for name, cls in classes if issubclass(cls, GlobalConstraint) and name != "GlobalConstraint"]
    classes = [(name, cls) for name, cls in classes if name not in EXCLUDE_GLOBAL.get(solver, {})]

    for name, cls in classes:
        if solver in EXCLUDE_GLOBAL and name in EXCLUDE_GLOBAL[solver]:
            continue

        if name == "Xor":
            yield cp.Xor(BOOL_ARGS)
            yield cp.Xor(BOOL_ARGS + [True,False])
            continue
        elif name == "Inverse":
            yield cp.Inverse(NUM_ARGS, [1,0,2])
        elif name == "Table":
            yield cp.Table(NUM_ARGS, [[0,1,2],[1,2,0],[1,0,2]])
            yield cp.Table(BOOL_ARGS, [[1,0,0],[0,1,0],[0,0,1]])
        elif name == "Regular":
            yield cp.Regular(cp.intvar(0,3, shape=3, name="x"), [("a", 1, "b"), ("b", 1, "c"), ("b", 0, "b"), ("c", 1, "c"), ("c", 0, "b")], "a", ["c"])
        elif name == "NegativeTable":
            yield cp.NegativeTable(NUM_ARGS, [[0, 1, 2], [1, 2, 0], [1, 0, 2]])
        elif name == "ShortTable":
            yield cp.ShortTable(NUM_ARGS, [[0,"*",2], ["*","*",1]])
        elif name == "IfThenElse":
            yield cp.IfThenElse(*BOOL_ARGS)
        elif name == "InDomain":
            yield cp.InDomain(NUM_VAR, [0,1,6])
        elif name == "Cumulative":
            s = cp.intvar(0, 10, shape=3, name="start")
            e = cp.intvar(0, 10, shape=3, name="end")
            dur = [1, 4, 3]
            demand = [4, 5, 7]
            cap = 10
            yield cp.Cumulative(s, dur, e, demand, cap)
            yield cp.all(cp.Cumulative(s, dur, e, demand, cap).decompose(how="time")[0])
            yield cp.all(cp.Cumulative(s, dur, e, demand, cap).decompose(how="task")[0])

            yield cp.Cumulative(start=s, duration=dur, demand=demand, capacity=cap) # also try with no end provided
            if solver != "pumpkin": # only supports with fixed durations
                yield cp.Cumulative(s.tolist()+[cp.intvar(0,10, name="start_2")], dur + [cp.intvar(-3,3, name="dur_2")], e.tolist()+[cp.intvar(0,10, name="end_2")], 1, cap)
                yield cp.Cumulative(s, dur, e, cp.intvar(-3,3,shape=3,name="demand"), cap)
            continue

        elif name == "CumulativeOptional":
            s = cp.intvar(0, 10, shape=4, name="start")
            e = cp.intvar(0, 10, shape=4, name="end")
            dur = [1, 4, 3, 2]
            demand = [11, 4, 8, 7]
            is_present = [cp.boolvar(name="start[0]_present"), cp.boolvar(name="start[1]_present"), True, False]
            cap = 10
            yield cls(s, dur, e, demand, cap, is_present)
        elif name == "GlobalCardinalityCount":
            vals = [1, 2, 3]
            cnts = cp.intvar(0,10,shape=3,name="vals")
            yield cp.GlobalCardinalityCount(NUM_ARGS, vals, cnts)
        elif name == "AllDifferentExceptN":
            vals = [1, 2, 3]
            yield cp.AllDifferentExceptN(NUM_ARGS, vals)
        elif name == "AllEqualExceptN":
            vals = [1, 2, 3]
            yield cp.AllEqualExceptN(NUM_ARGS, vals)
        elif name == "Precedence":
            x = cp.intvar(0,5, shape=3, name="x")
            yield cp.Precedence(x, [3,1,0])
        elif name == "NoOverlap":
            s = cp.intvar(0, 10, shape=3, name="start")
            e = cp.intvar(0, 10, shape=3, name="end")
            dur = [1,4,3]
            yield cp.NoOverlap(s, dur, e)
            yield cp.NoOverlap(s, dur)
            if solver != "pumpkin": # only supports with fixed durations
                yield cp.NoOverlap(s.tolist()+[cp.intvar(0,10)], dur + [cp.intvar(-3,3)], e.tolist()+[cp.intvar(0,10)])
            continue
        elif name == "NoOverlapOptional":
            s = cp.intvar(0, 10, shape=4, name="start")
            e = cp.intvar(0, 10, shape=4, name="end")
            dur = [1, 4, 3, 2]
            is_present = [cp.boolvar(name="start[0]_present"), cp.boolvar(name="start[1]_present"), True, False]
            yield cls(s, dur, e, is_present)
        elif name == "GlobalCardinalityCount":
            vals = [1, 2, 3]
            cnts = cp.intvar(0,10,shape=3, name="counts")
            yield cp.GlobalCardinalityCount(NUM_ARGS, vals, cnts)
        elif name == "LexLessEq":
            X = cp.intvar(0, 3, shape=3, name="X")
            Y = cp.intvar(0, 3, shape=3, name="Y")
            yield cp.LexLessEq(X, Y)
        elif name == "LexLess":
            X = cp.intvar(0, 3, shape=3, name="X")
            Y = cp.intvar(0, 3, shape=3, name="Y")
            yield cp.LexLess(X, Y)
        elif name == "LexChainLess":
            X = cp.intvar(0, 3, shape=(3,3), name="X")
            yield cp.LexChainLess(X)
        elif name == "LexChainLessEq":
            X = cp.intvar(0, 3, shape=(3,3), name="X")
            yield cp.LexChainLessEq(X)
        else: # default constructor, list of numvars
            yield cls(NUM_ARGS)            


# also global functions
def global_functions(solver):
    """
        Generate all global functions
    """
    classes = inspect.getmembers(cp.expressions.globalfunctions, inspect.isclass)
    classes = [(name, cls) for name, cls in classes if issubclass(cls, GlobalFunction) and name != "GlobalFunction"]
    classes = [(name, cls) for name, cls in classes if name not in EXCLUDE_GLOBAL.get(solver, {})]

    for name, cls in classes:
        if solver in EXCLUDE_GLOBAL and name in EXCLUDE_GLOBAL[solver]:
            continue
        
        if name == "Abs":
            yield cp.Abs(NUM_ARGS[0])
        elif name == "Count":
            yield cp.Count(NUM_ARGS, NUM_VAR)
        elif name == "Element":
            yield cp.Element(NUM_ARGS, POS_VAR)
        elif name == "NValueExcept":
            yield cp.NValueExcept(NUM_ARGS, 3)
        elif name == "Among":
            yield cp.Among(NUM_ARGS, [1,2])
        elif name == "Division":
            yield cp.Division(NUM_ARGS[0], NUM_ARGS[1])
        elif name == "Modulo":
            yield cp.Modulo(NUM_ARGS[0], NUM_ARGS[1])
        elif name == "Power":
            yield cp.Power(SMALL_NUM_ARG[0], 3)
        elif name == "Multiplication":
            yield cp.Multiplication(NUM_ARGS[0], NUM_ARGS[1])
            yield cp.Multiplication(BOOL_ARGS[0], BOOL_ARGS[1])
            yield cp.Multiplication(3, BOOL_ARGS[0])
            yield cp.Multiplication(3, NUM_ARGS[0])

            if solver != "minizinc":  # bug in minizinc, see https://github.com/MiniZinc/libminizinc/issues/962
                yield cp.Multiplication(3, BOOL_ARGS[0])
        else:
            yield cls(NUM_ARGS)

def generate_cases(solver):
    yield cp.boolvar(name="x") >= 0  # issue #736
    x, y = cp.intvar(1, 3,shape=2, name=["x", "y"])
    yield x ** 2 - 2*x*y + y**2 <= 3

    p, q, r = [cp.boolvar(name=x) for x in "pqr"]
    yield p | q
    yield r | (p & q)
    # yield z | (p & q)
    # yield ((cp.boolvar(name="x") >= 0) | (cp.boolvar(name="y") >= 0))  # issue #736

def reify_imply_exprs(solver):
    """
    - Reification (double implication): Boolexpr == Var    (CPMpy class 'Comparison')
    - Implication: Boolexpr -> Var                         (CPMpy class 'Operator', is_bool())
                   Var -> Boolexpr                         (CPMpy class 'Operator', is_bool())
    """
    for bool_expr in bool_exprs(solver):
        yield bool_expr.implies(BOOL_VAR)
        yield BOOL_VAR.implies(bool_expr)
        yield bool_expr == BOOL_VAR

    for comp_expr in comp_constraints(solver):
        lhs, rhs = comp_expr.args
        yield comp_expr.implies(BOOL_VAR)
        yield BOOL_VAR.implies(comp_expr)
        yield comp_expr == BOOL_VAR


def verify(constraint):
    vars_ = get_variables(constraint)
    assignment = {v.name: v.value() for v in sorted(vars_, key=lambda v: v.name)}
    assert all(val is not None for val in assignment.values()), "Expected all variables to be assigned"
    assert argval(constraint), f"argval failed for {constraint} with assignment {assignment}"
    assert constraint.value(), f"value() failed for {constraint} with assignment {assignment}"

def all_constraints(solver):
    """Combined generator for all constraint types, yielding (id, constraint) tuples."""
    generators = [
        ("bool", bool_exprs),
        ("comp", comp_constraints),
        ("case", generate_cases),
        ("reify", reify_imply_exprs),
    ]
    idx = 0
    for prefix, gen in generators:
        for i, c in enumerate(gen(solver)):
            yield (f"{idx}-{prefix}_{i}", c)
            idx += 1

@pytest.mark.generate_constraints.with_args(all_constraints)
@skip_on_missing_pblib(skip_on_exception_only=True)
def test_constraints(solver, constraint):
    """
        Tests constraint by posting it to the solver and checking the value after solve.
    """

    # Show model with variables and their domains
    model = Model(constraint)
    vars_ = get_variables(constraint)
    var_info = {v.name: (v.lb, v.ub) for v in sorted(vars_, key=lambda v: v.name)}
    print(f"Model: {model}")
    print(f"Variables: {var_info}")

    if ALL_SOLS:
        n_sols = SolverLookup.get(solver, model).solveAll(display=lambda: verify(constraint), solution_limit=100)
        assert n_sols >= 1, f"Unexpected unsat: {constraint}"
    else:
        # print("TF", SolverLookup.get(solver).transform(model.constraints))
        assert SolverLookup.get(solver, model).solve(), f"Unexpected unsat: {constraint}"
        for constraint in model.constraints:
            verify(constraint)
        assert constraint.value()

if __name__ == "__main__":
    solver = None  # Use None for no solver-specific exclusions

    generators = [
        ("Boolean expressions", bool_exprs),
        ("Comparison constraints", comp_constraints),
        ("Global constraints", global_constraints),
        ("Global functions", global_functions),
        ("Reify/imply expressions", reify_imply_exprs),
    ]

    for name, gen in generators:
        print(f"\n{'='*60}")
        print(f"{name}")
        print('='*60)
        for i, expr in enumerate(gen(solver)):
            model = Model(expr)
            print(f"{i+1}. {model}")

