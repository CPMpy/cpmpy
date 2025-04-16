import inspect

import cpmpy
from cpmpy import Model, SolverLookup, BoolVal
from cpmpy.expressions.globalconstraints import *
from cpmpy.expressions.globalfunctions import *
from cpmpy.expressions.core import Comparison

import pytest

# CHANGE THIS if you want test a different solver
#   make sure that `SolverLookup.get(solver)` works
# also add exclusions to the 3 EXCLUDE_* below as needed
SOLVERNAMES = [name for name, solver in SolverLookup.base_solvers() if solver.supported()]
ALL_SOLS = False # test wheter all solutions returned by the solver satisfy the constraint

# Exclude some global constraints for solvers
NUM_GLOBAL = {
    "AllEqual", "AllDifferent", "AllDifferentExcept0",
    "AllDifferentExceptN", "AllEqualExceptN",
    "GlobalCardinalityCount", "InDomain", "Inverse", "Table", 'NegativeTable', "ShortTable", "Circuit",
    "Increasing", "IncreasingStrict", "Decreasing", "DecreasingStrict", 
    "Precedence", "Cumulative", "NoOverlap",
    "LexLess", "LexLessEq", "LexChainLess", "LexChainLessEq",
    # also global functions
    "Abs", "Element", "Minimum", "Maximum", "Count", "Among", "NValue", "NValueExcept"
}

# Solvers not supporting arithmetic constraints
SAT_SOLVERS = {"pysat", "pysdd"}

EXCLUDE_GLOBAL = {"pysat": NUM_GLOBAL,
                  "pysdd": NUM_GLOBAL | {"Xor"},
                  "z3": {"Inverse"},
                  "choco": {"Inverse"},
                  "ortools":{"Inverse"},
                  "exact": {"Inverse"},
                  "minizinc": {"IncreasingStrict"}, # bug #813 reported on libminizinc
                  "gcs": {}
                  }

# Exclude certain operators for solvers.
# Not all solvers support all operators in CPMpy
EXCLUDE_OPERATORS = {"gurobi": {},
                     "pysat": {"sum", "wsum", "sub", "mod", "div", "pow", "abs", "mul","-"},
                     "pysdd": {"sum", "wsum", "sub", "mod", "div", "pow", "abs", "mul","-"},
                     "exact": {},
                     }

# Variables to use in the rest of the test script
NUM_ARGS = [intvar(-3, 5, name=n) for n in "xyz"]   # Numerical variables
NN_VAR = intvar(0, 10, name="n_neg")                # Non-negative variable, needed in power functions
POS_VAR = intvar(1,10, name="s_pos")                # A strictly positive variable
NUM_VAR = intvar(0, 10, name="l")                   # A numerical variable

BOOL_ARGS = [boolvar(name=n) for n in "abc"]        # Boolean variables
BOOL_VAR = boolvar(name="p")                        # A boolean variable

def _generate_inputs(generator):
    exprs = []
    for solver in SOLVERNAMES:
        exprs += [(solver, expr) for expr in generator(solver)]
    return exprs


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
            operator_args = [list(range(len(NUM_ARGS))), NUM_ARGS]
        elif name == "div" or name == "pow":
            operator_args = [NN_VAR,3]
        elif arity != 0:
            operator_args = NUM_ARGS[:arity]
        else:
            operator_args = NUM_ARGS

        yield Operator(name, operator_args)

    # boolexprs are also numeric
    for expr in bool_exprs(solver):
        yield expr

    # also global functions
    classes = inspect.getmembers(cpmpy.expressions.globalfunctions, inspect.isclass)
    classes = [(name, cls) for name, cls in classes if issubclass(cls, GlobalFunction) and name != "GlobalFunction"]
    classes = [(name, cls) for name, cls in classes if name not in EXCLUDE_GLOBAL.get(solver, {})]

    for name, cls in classes:
        if name == "Abs":
            expr = cls(NUM_ARGS[0])
        elif name == "Count":
            expr = cls(NUM_ARGS, NUM_VAR)
        elif name == "Element":
            expr = cls(NUM_ARGS, POS_VAR)
        elif name == "NValueExcept":
            expr = cls(NUM_ARGS, 3)
        elif name == "Among":
            expr = cls(NUM_ARGS, [1,2])
        else:
            expr = cls(NUM_ARGS)

        if solver in EXCLUDE_GLOBAL and expr.name in EXCLUDE_GLOBAL[solver]:
            continue
        else:
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

    for cpm_cons in global_constraints(solver):
        yield cpm_cons

def global_constraints(solver):
    """
        Generate all global constraints
        -  AllDifferent, AllEqual, Circuit,  Minimum, Maximum, Element,
           Xor, Cumulative, NValue, Count, Increasing, Decreasing, IncreasingStrict, DecreasingStrict, LexLessEq, LexLess
    """
    classes = inspect.getmembers(cpmpy.expressions.globalconstraints, inspect.isclass)
    classes = [(name, cls) for name, cls in classes if issubclass(cls, GlobalConstraint) and name != "GlobalConstraint"]
    classes = [(name, cls) for name, cls in classes if name not in EXCLUDE_GLOBAL.get(solver, {})]

    for name, cls in classes:
        if solver in EXCLUDE_GLOBAL and name in EXCLUDE_GLOBAL[solver]:
            continue

        if name == "Xor":
            expr = cls(BOOL_ARGS)
        elif name == "Inverse":
            expr = cls(NUM_ARGS, [1,0,2])
        elif name == "Table":
            expr = cls(NUM_ARGS, [[0,1,2],[1,2,0],[1,0,2]])
        elif name == "NegativeTable":
            expr = cls(NUM_ARGS, [[0, 1, 2], [1, 2, 0], [1, 0, 2]])
        elif name == "ShortTable":
            expr = cls(NUM_ARGS, [[0,"*",2], ["*","*",1]])
        elif name == "IfThenElse":
            expr = cls(*BOOL_ARGS)
        elif name == "InDomain":
            expr = cls(NUM_VAR, [0,1,6])
        elif name == "Cumulative":
            s = intvar(0, 10, shape=3, name="start")
            e = intvar(0, 10, shape=3, name="end")
            dur = [1, 4, 3]
            demand = [4, 5, 7]
            cap = 10
            expr = Cumulative(s, dur, e, demand, cap)
        elif name == "GlobalCardinalityCount":
            vals = [1, 2, 3]
            cnts = intvar(0,10,shape=3)
            expr = cls(NUM_ARGS, vals, cnts)
        elif name == "AllDifferentExceptN":
            expr = cls(NUM_ARGS, NUM_VAR)
        elif name == "AllEqualExceptN":
            expr = cls(NUM_ARGS, NUM_VAR)
        elif name == "Precedence":
            x = intvar(0,5, shape=3, name="x")
            expr = cls(x, [3,1,0])
        elif name == "NoOverlap":
            s = intvar(0, 10, shape=3, name="start")
            e = intvar(0, 10, shape=3, name="end")
            dur = [1,4,3]
            expr = cls(s, dur, e)
        elif name == "GlobalCardinalityCount":
            vals = [1, 2, 3]
            cnts = intvar(0,10,shape=3)
            expr = cls(NUM_ARGS, vals, cnts)
        elif name == "LexLessEq":
            X = intvar(0, 3, shape=3)
            Y = intvar(0, 3, shape=3)
            expr = LexLessEq(X, Y)
        elif name == "LexLess":
            X = intvar(0, 3, shape=3)
            Y = intvar(0, 3, shape=3)
            expr = LexLess(X, Y)
        elif name == "LexChainLess":
            X = intvar(0, 3, shape=(3,3))
            expr = LexChainLess(X)
        elif name == "LexChainLessEq":
            X = intvar(0, 3, shape=(3,3))
            expr = LexChainLess(X)
        else: # default constructor, list of numvars
            expr= cls(NUM_ARGS)            

        if solver in EXCLUDE_GLOBAL and name in EXCLUDE_GLOBAL[solver]:
            continue
        else:
            yield expr


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


def verify(cons):
    assert argval(cons)
    assert cons.value()


@pytest.mark.parametrize(("solver","constraint"),list(_generate_inputs(bool_exprs)), ids=str)
def test_bool_constraints(solver, constraint):
    """
        Tests boolean constraint by posting it to the solver and checking the value after solve.
    """
    if ALL_SOLS:
        n_sols = SolverLookup.get(solver, Model(constraint)).solveAll(display=lambda: verify(constraint))
        assert n_sols >= 1
    else:
        assert SolverLookup.get(solver, Model(constraint)).solve()
        assert argval(constraint)
        assert constraint.value()


@pytest.mark.parametrize(("solver","constraint"), list(_generate_inputs(comp_constraints)),  ids=str)
def test_comparison_constraints(solver, constraint):
    """
        Tests comparison constraint by posting it to the solver and checking the value after solve.
    """
    if ALL_SOLS:
        n_sols = SolverLookup.get(solver, Model(constraint)).solveAll(display= lambda: verify(constraint))
        assert n_sols >= 1
    else:
        assert SolverLookup.get(solver,Model(constraint)).solve()
        assert argval(constraint)
        assert constraint.value()


@pytest.mark.parametrize(("solver","constraint"), list(_generate_inputs(reify_imply_exprs)),  ids=str)
def test_reify_imply_constraints(solver, constraint):
    """
        Tests boolean expression by posting it to solver and checking the value after solve.
    """
    if ALL_SOLS:
        n_sols = SolverLookup.get(solver, Model(constraint)).solveAll(display=lambda: verify(constraint))
        assert n_sols >= 1
    else:
        assert SolverLookup.get(solver, Model(constraint)).solve()
        assert argval(constraint)
        assert constraint.value()
