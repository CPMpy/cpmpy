from collections import namedtuple
from types import SimpleNamespace

import cpmpy as cp

from cpmpy.tools.io.scip import _load_scip_objective


_SCIPVar = namedtuple("_SCIPVar", ["name"])
_SCIPTerm = namedtuple("_SCIPTerm", ["vartuple"])


def _scip_term(*names):
    return _SCIPTerm(tuple(_SCIPVar(name) for name in names))


def test_scip_objective_supports_higher_order_terms():
    x = cp.intvar(0, 2, name="x")
    y = cp.intvar(0, 2, name="y")
    z = cp.intvar(0, 2, name="z")
    scip_objective = SimpleNamespace(
        terms={
            _scip_term("x"): 2,
            _scip_term("x", "y"): 3,
            _scip_term("x", "y", "z"): -4,
        }
    )

    obj = _load_scip_objective(
        scip_objective,
        {"x": x, "y": y, "z": z},
        assume_integer=False,
    )

    assert str(obj) == "sum([2, 3, -4] * [x, (x) * (y), ((x) * (y)) * (z)])"


def test_scip_objective_supports_repeated_variables_in_terms():
    x = cp.intvar(0, 3, name="x")
    scip_objective = SimpleNamespace(terms={_scip_term("x", "x"): 5})

    obj = _load_scip_objective(scip_objective, {"x": x}, assume_integer=False)

    assert str(obj) == "5 * ((x) * (x))"
