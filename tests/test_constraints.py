import unittest

from cpmpy import boolvar, intvar, Model
from cpmpy.expressions.core import Comparison, Operator
from cpmpy.solvers import CPM_gurobi, CPM_ortools, CPM_minizinc

import pytest

SOLVER_CLASS = CPM_gurobi

EXCLUDE_MAP = {CPM_ortools : ("sub","div","mod","pow"),
               CPM_gurobi : ("sub", "mod")}


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


@pytest.mark.parametrize("cname", Comparison.allowed)
def test_comp_constraints(cname):
    # Integer variables
    i, j, k = [intvar(0, 3, name=n) for n in "ijk"]

    SOLVER_CLASS(Model(Comparison(cname, i, j))).solve()
    string = f"{i.value()} {cname} {j.value()}"
    assert eval(string)


COMBOS = [(op, comp) for comp in Comparison.allowed for op in Operator.allowed]


@pytest.mark.parametrize("o_name,c_name", COMBOS)
def test_operator_comp_constraints(o_name, c_name):
    """
        Test all flattened expressions.
        See cpmpy/transformations/flatten_model
    """

    if o_name in EXCLUDE_MAP[SOLVER_CLASS] or c_name in EXCLUDE_MAP[SOLVER_CLASS]:
        return

    # Integer variables
    i, j = [intvar(-3, 3, name=n) for n in "ij"]
    k, l = [intvar(0, 3, name=n) for n in "kl"]

    a,b,c = [boolvar(name=n) for n in "abc"]

    arity, is_bool = Operator.allowed[o_name]
    if is_bool:
        # Can never be the outcome of flatten. See /tranformations/flatten_model
        return
    if o_name == "wsum":
        args = [[1, 2, 3], [i, j, k]]
    elif arity == 1:
        args = [i]
    elif arity == 2:
        args = [k, 2] if o_name in ("div", "pow") else [i,k]
    else:
        args = [a,b,c] if is_bool else [i,j,k]

    constraint = Comparison(c_name, Operator(o_name, args), l)
    SOLVER_CLASS(Model(constraint)).solve()
    assert constraint.value()

class test_reify_contraints(unittest.TestCase):

    def setUp(self) -> None:
        self.a,self.b,self.c = [boolvar(name=n) for n in "abc"]
        self.solver = SOLVER_CLASS()

    def test_and(self):
        constr = self.a & self.b == self.c
        self.solver += constr
        assert self.solver.solve()
        self.assertTrue(constr.value())
