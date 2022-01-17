import unittest

from cpmpy import boolvar, intvar, Model
from cpmpy.expressions.core import Comparison, Operator
from cpmpy.solvers import CPM_gurobi, CPM_ortools, CPM_minizinc

import pytest

SOLVER_CLASS = CPM_ortools

# Exclude certain operators for solvers.
# Not all solvers support all operators in CPMpy
EXCLUDE_MAP = {CPM_ortools: ("sub", "div", "mod", "pow"),
               CPM_gurobi: ("sub", "mod")}

# Variables to use in the rest of the test script
NUM_ARGS = [intvar(-3, 3, name=n) for n in "xyz"]   # Numerical variables
NUM_VAR = intvar(-3,3, name="l")                    # A numerical variable

BOOL_ARGS = [boolvar(name=n) for n in "abc"]        # Boolean variables
BOOL_VAR = boolvar(name="p")                        # A boolean variable


def test_base_constraints():
    # Bool variables
    x, y, z = [boolvar(name=n) for n in "xyz"]

    # Test var
    SOLVER_CLASS(Model(x)).solve()
    assert x.value()

    # Test and
    SOLVER_CLASS(Model(Operator("and", [x, y, z]))).solve()
    assert (x.value() & y.value() & z.value())

    # Test or
    SOLVER_CLASS(Model(Operator("or", [x, y, z]))).solve()
    assert (x.value() | y.value() | z.value())

    # Test xor
    SOLVER_CLASS(Model(Operator("xor", [x, y, z]))).solve()
    assert (x.value() ^ y.value() ^ z.value())

    # Test implies
    SOLVER_CLASS(Model(x.implies(y))).solve()
    assert (~x.value() | y.value())

    # Test eq
    SOLVER_CLASS(Model(x == y)).solve()
    assert (x.value() == y.value())

    # Test neq
    SOLVER_CLASS(Model(x != y)).solve()
    assert (x.value() != y.value())


def numexprs():
    """
    Generate all numerical expressions
    Numexpr:
        - Operator (non-Boolean) with all args Var/constant (examples: +,*,/,mod,wsum)
                                                           (CPMpy class 'Operator', not is_bool())
        - Global constraint (non-Boolean) (examples: Max,Min,Element)
                                                           (CPMpy class 'GlobalConstraint', not is_bool()))
    """
    names = [(name, arity) for name, (arity, is_bool) in Operator.allowed.items() if not is_bool]
    names = [(name, arity) for name, arity in names if name not in EXCLUDE_MAP[SOLVER_CLASS]]
    for name, arity in names:
        if name == "wsum":
            operator_args = [list(range(len(NUM_ARGS))), NUM_ARGS]
        elif arity != 0:
            operator_args = NUM_ARGS[:arity]
        else:
            operator_args = NUM_ARGS

        yield Operator(name, operator_args)


# Generate all possible comparison constraints
def comp_constraints():
    """
        Generate all comparison constraints
        - Numeric equality:  Numexpr == Var                (CPMpy class 'Comparison')
                         Numexpr == Constant               (CPMpy class 'Comparison')
        - Numeric disequality: Numexpr != Var              (CPMpy class 'Comparison')
                           Numexpr != Constant             (CPMpy class 'Comparison')
        - Numeric inequality (>=,>,<,<=): Numexpr >=< Var  (CPMpy class 'Comparison')
    """
    for comp_name in Comparison.allowed:
        for numexpr in numexprs():
            for rhs in [NUM_VAR, 1]:
                yield Comparison(comp_name, numexpr, rhs)


# Generate all possible boolean expressions
def bool_exprs():
    """
        Generate all boolean expressions:
         - Boolean operators: and([Var]), or([Var]), xor([Var]) (CPMpy class 'Operator', is_bool())
        - Boolean equality: Var == Var                          (CPMpy class 'Comparison')
    """
    names = [(name, arity) for name, (arity, is_bool) in Operator.allowed.items() if is_bool]
    names = [(name, arity) for name, arity in names if name not in EXCLUDE_MAP[SOLVER_CLASS]]

    for name, arity in names:
        if arity != 0:
            operator_args = BOOL_ARGS[:arity]
        else:
            operator_args = BOOL_ARGS

        yield Operator(name, operator_args)

    for eq_name in ["==", "!="]:
        yield Comparison(eq_name, *BOOL_ARGS[:2])

    # TODO global constraints

# Generate all reify/imply constraints
def reify_imply_exprs():
    """
    - Reification (double implication): Boolexpr == Var    (CPMpy class 'Comparison')
    - Implication: Boolexpr -> Var                         (CPMpy class 'Operator', is_bool())
                   Var -> Boolexpr                         (CPMpy class 'Operator', is_bool())
    """

    for bool_expr in bool_exprs():
        yield bool_expr == BOOL_VAR
        yield bool_expr.implies(BOOL_VAR)
        yield BOOL_VAR.implies(bool_expr)

    for comp_expr in comp_constraints():
        yield comp_expr == BOOL_VAR
        yield comp_expr.implies(BOOL_VAR)
        yield BOOL_VAR.implies(comp_expr)


@pytest.mark.parametrize("constraint", comp_constraints(), ids=lambda c: str(c))
def test_comparisons(constraint):
    """
        Tests comparison constraint by posting it to the solver and checking the value after solve.
    """
    assert SOLVER_CLASS(Model(constraint)).solve()
    assert constraint.value()


@pytest.mark.parametrize("constraint", reify_imply_exprs(), ids=lambda c: str(c))
def test_reify_imply(constraint):
    """
        Tests boolean expression by posting it to solver and checking the value after solve.
    """
    assert SOLVER_CLASS(Model(constraint)).solve()
    assert constraint.value()