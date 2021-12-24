from cpmpy import boolvar, intvar, Model
from cpmpy.expressions.core import Comparison, Operator
from cpmpy.solvers import CPM_gurobi

import pytest

SOLVER_CLASS = CPM_gurobi

def test_base_constraints(self):
    # Bool variables
    x, y, z = [boolvar(name=n) for n in "xyz"]

    # Test var
    SOLVER_CLASS(Model(x)).solve()
    self.assertTrue(x.value())

    # Test and
    SOLVER_CLASS(Model(Operator("and", [x, y, z]))).solve()
    self.assertTrue(x.value() & y.value() & z.value())

    # Test or
    SOLVER_CLASS(Model(Operator("or", [x, y, z]))).solve()
    self.assertTrue(x.value() | y.value() | z.value())

    # Test xor
    SOLVER_CLASS(Model(Operator("xor", [x, y, z]))).solve()
    self.assertTrue(x.value() ^ y.value() ^ z.value())

    # Test implies
    SOLVER_CLASS(Model(x.implies(y))).solve()
    self.assertTrue(~x.value() | y.value())

    # Test eq
    SOLVER_CLASS(x == y)
    self.assertTrue(x.value() == y.value())

    # Test neq
    SOLVER_CLASS(x != y)
    self.assertTrue(x.value() != y.value())


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
        Tests all allowed combinations of operators and combinations
    """

    # Integer variables
    i, j, k, l = [intvar(0, 3, name=n) for n in "ijkl"]

    is_bool, arity = Operator.allowed[o_name]
    if is_bool:
        return

    infix = o_name in Operator.printmap

    if arity == 1:
        SOLVER_CLASS(Model(Comparison(c_name, Operator(o_name, i), l))).solve()
        if infix:
            string = f"{Operator.printmap[o_name]} {i.value()} {c_name} {l.value()}"
        else:
            string = f"{o_name}({i.value()}) {c_name} {l.value()}"

    elif o_name == "wsum":
        args = [[1, 2, 3], [i, j, k]]
        SOLVER_CLASS(Model(Comparison(c_name, Operator(o_name, args), l))).solve()
        string = f"{sum([a * b.value() for a, b in zip(args[0], args[1])])} {c_name} {l.value()}"

    else:
        args = [i, j] if arity == 2 else [i, j, k]
        SOLVER_CLASS(Model(Comparison(c_name, Operator(o_name, args), l))).solve()
        if infix:
            string = Operator.printmap[o_name].join([str(a.value()) for a in args]) + f" {c_name} {l.value()}"
        else:
            string = f"{o_name}({[a.value() for a in args]}) {c_name} {l.value()}"

    assert eval(string)
