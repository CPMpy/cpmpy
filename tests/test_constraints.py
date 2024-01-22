from cpmpy import Model, SolverLookup, BoolVal
from cpmpy.expressions.globalconstraints import *
from cpmpy.expressions.globalfunctions import *

import pytest

# CHANGE THIS if you want test a different solver
#   make sure that `SolverLookup.get(solver)` works
# also add exclusions to the 3 EXCLUDE_* below as needed
SOLVERNAMES = [name for name, solver in SolverLookup.base_solvers() if solver.supported()]

# Exclude some global constraints for solvers
# Can be used when .value() method is not implemented/contains bugs
EXCLUDE_GLOBAL = {"ortools": {},
                  "gurobi": {},
                  "minizinc": {"circuit"},
                  "pysat": {"circuit", "element","min","max","count", "nvalue", "allequal","alldifferent","cumulative"},
                  "pysdd": {"circuit", "element","min","max","count", "nvalue", "allequal","alldifferent","cumulative",'xor'},
                  "exact": {},
                  }

# Exclude certain operators for solvers.
# Not all solvers support all operators in CPMpy
EXCLUDE_OPERATORS = {"gurobi": {"mod"},
                     "pysat": {"sum", "wsum", "sub", "mod", "div", "pow", "abs", "mul","-"},
                     "pysdd": {"sum", "wsum", "sub", "mod", "div", "pow", "abs", "mul","-"},
                     "exact": {"mod","pow","div","mul"},
                     }

# Some solvers only support a subset of operators in imply-constraints
# This subset can differ between left and right hand side of the implication
EXCLUDE_IMPL = {"ortools": {},
                "minizinc": {},
                "z3": {},
                "pysat": {},
                "pysdd": {},
                "exact": {"mod","pow","div","mul"},
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
        - Global constraint (non-Boolean) (examples: Max,Min,Element)
                                                           (CPMpy class 'GlobalConstraint', not is_bool()))
    """
    names = [(name, arity) for name, (arity, is_bool) in Operator.allowed.items() if not is_bool]
    if solver in EXCLUDE_OPERATORS:
        names = [(name, arity) for name, arity in names if name not in EXCLUDE_OPERATORS[solver]]
    for name, arity in names:
        if name == "wsum":
            operator_args = [list(range(len(NUM_ARGS))), NUM_ARGS]
        elif name == "div" or name == "pow":
            operator_args = [NN_VAR,2]
        elif name == "mod":
            operator_args = [NN_VAR,POS_VAR]
        elif arity != 0:
            operator_args = NUM_ARGS[:arity]
        else:
            operator_args = NUM_ARGS

        yield Operator(name, operator_args)


# Generate all possible comparison constraints
def comp_constraints(solver):
    """
        Generate all comparison constraints
        - Numeric equality:  Numexpr == Var                (CPMpy class 'Comparison')
                         Numexpr == Constant               (CPMpy class 'Comparison')
        - Numeric disequality: Numexpr != Var              (CPMpy class 'Comparison')
                           Numexpr != Constant             (CPMpy class 'Comparison')
        - Numeric inequality (>=,>,<,<=): Numexpr >=< Var  (CPMpy class 'Comparison')
    """
    for comp_name in Comparison.allowed:
        for numexpr in numexprs(solver):
            for rhs in [NUM_VAR, BOOL_VAR, 1, BoolVal(True)]:
                yield Comparison(comp_name, numexpr, rhs)

    for comp_name in Comparison.allowed:
        for glob_expr in global_constraints(solver):
            if not glob_expr.is_bool():
                for rhs in [NUM_VAR, BOOL_VAR, 1, BoolVal(True)]:
                    if comp_name == "<" and get_bounds(glob_expr)[0] >= get_bounds(rhs)[1]:
                        continue
                    yield Comparison(comp_name, glob_expr, rhs)

    if solver == "z3":
        for comp_name in Comparison.allowed:
            for boolexpr in bool_exprs(solver):
                for rhs in [NUM_VAR, BOOL_VAR, 1, BoolVal(True)]:
                    if comp_name == '>':
                        # >1 is unsat for boolean expressions, so change it to 0
                        if isinstance(rhs, int) and rhs == 1:
                            rhs = 0
                        if isinstance(rhs, BoolVal) and rhs.args[0] == True:
                            rhs = BoolVal(False)
                    yield Comparison(comp_name, boolexpr, rhs)


# Generate all possible boolean expressions
def bool_exprs(solver):
    """
        Generate all boolean expressions:
        - Boolean operators: and([Var]), or([Var])              (CPMpy class 'Operator', is_bool())
        - Boolean equality: Var == Var                          (CPMpy class 'Comparison')
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
        if cpm_cons.is_bool():
            yield cpm_cons

def global_constraints(solver):
    """
        Generate all global constraints
        -  AllDifferent, AllEqual, Circuit,  Minimum, Maximum, Element,
           Xor, Cumulative, NValue, Count
    """
    global_cons = [AllDifferent, AllEqual, Minimum, Maximum, NValue]
    for global_type in global_cons:
        cons = global_type(NUM_ARGS)
        if solver not in EXCLUDE_GLOBAL or cons.name not in EXCLUDE_GLOBAL[solver]:
            yield cons

    # "special" constructors
    if solver not in EXCLUDE_GLOBAL or "element" not in EXCLUDE_GLOBAL[solver]:
        yield cpm_array(NUM_ARGS)[NUM_VAR]

    if solver not in EXCLUDE_GLOBAL or "xor" not in EXCLUDE_GLOBAL[solver]:
        yield Xor(BOOL_ARGS)

    if solver not in EXCLUDE_GLOBAL or "count" not in EXCLUDE_GLOBAL[solver]:
        yield Count(NUM_ARGS, NUM_VAR)

    if solver not in EXCLUDE_GLOBAL or "cumulative" not in EXCLUDE_GLOBAL[solver]:
        s = intvar(0,10,shape=3,name="start")
        e = intvar(0,10,shape=3,name="end")
        dur = [1,4,3]
        demand = [4,5,7]
        cap = 10
        yield Cumulative(s, dur, e, demand, cap)


def reify_imply_exprs(solver):
    """
    - Reification (double implication): Boolexpr == Var    (CPMpy class 'Comparison')
    - Implication: Boolexpr -> Var                         (CPMpy class 'Operator', is_bool())
                   Var -> Boolexpr                         (CPMpy class 'Operator', is_bool())
    """
    for bool_expr in bool_exprs(solver):
        if solver not in EXCLUDE_IMPL or  \
                bool_expr.name not in EXCLUDE_IMPL[solver]:
            yield bool_expr.implies(BOOL_VAR)
            yield BOOL_VAR.implies(bool_expr)
            yield bool_expr == BOOL_VAR

    for comp_expr in comp_constraints(solver):
        lhs, rhs = comp_expr.args
        if solver not in EXCLUDE_IMPL or \
                (not isinstance(lhs, Expression) or lhs.name not in EXCLUDE_IMPL[solver]) and \
                (not isinstance(rhs, Expression) or rhs.name not in EXCLUDE_IMPL[solver]):
            yield comp_expr.implies(BOOL_VAR)
            yield BOOL_VAR.implies(comp_expr)
            yield comp_expr == BOOL_VAR


@pytest.mark.parametrize(("solver","constraint"),_generate_inputs(bool_exprs), ids=str)
def test_bool_constaints(solver, constraint):
    """
        Tests boolean constraint by posting it to the solver and checking the value after solve.
    """
    assert SolverLookup.get(solver, Model(constraint)).solve()
    assert constraint.value()


@pytest.mark.parametrize(("solver","constraint"), _generate_inputs(comp_constraints),  ids=str)
def test_comparison_constraints(solver, constraint):
    """
        Tests comparison constraint by posting it to the solver and checking the value after solve.
    """
    assert SolverLookup.get(solver,Model(constraint)).solve()
    assert constraint.value()


@pytest.mark.parametrize(("solver","constraint"), _generate_inputs(reify_imply_exprs),  ids=str)
def test_reify_imply_constraints(solver, constraint):
    """
        Tests boolean expression by posting it to solver and checking the value after solve.
    """
    assert SolverLookup.get(solver, Model(constraint)).solve()
    assert constraint.value()
