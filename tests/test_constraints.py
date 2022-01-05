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

    arity, is_bool = Operator.allowed[o_name]
    if is_bool:
        return

    eval_map = {key: val for key, val in Operator.printmap.items()}
    eval_map.update({"mod": "%"})
    infix = o_name in eval_map


    if o_name == "wsum":
        args = [[1, 2, 3], [i, j, k]]
        constraint = Comparison(c_name, Operator(o_name, args), l)
        SOLVER_CLASS(Model(constraint)).solve()
        string = f"{sum([a * b.value() for a, b in zip(args[0], args[1])])} {c_name} {l.value()}"

    elif o_name == "->":
        args = [i, j]
        constraint = Comparison(c_name, Operator(o_name, args), l)
        SOLVER_CLASS(Model(constraint)).solve()
        string = f"not {i.value()} or {j.value()} {c_name} {l.value()}"

    elif arity == 1:
        constraint = Comparison(c_name, Operator(o_name, [i]), l)
        SOLVER_CLASS(Model(constraint)).solve()
        string = f"{o_name}({i.value()}) {c_name} {l.value()}"

    elif arity == 2:
        args = [k, 2] if o_name in ("div", "pow") else [i,k]
        constraint = Comparison(c_name, Operator(o_name, args), l)
        SOLVER_CLASS(Model(constraint)).solve()
        if infix:
            string = f"({args[0]} {eval_map[o_name]} {args[1]}) {c_name} {l.value()}"
        else:
            string = f"{o_name}({args[0]},{args[1]}) {c_name} {l.value()}"

    else:
        args = [i, j, k]
        constraint = Comparison(c_name, Operator(o_name, args), l)
        SOLVER_CLASS(Model(constraint)).solve()
        if infix:
            string = eval_map[o_name].join([str(a.value()) for a in args]) + f" {c_name} {l.value()}"
        else:
            string = f"{o_name}({[a.value() for a in args]}) {c_name} {l.value()}"

    print(string)
    assert eval(string)
