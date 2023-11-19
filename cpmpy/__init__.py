"""
    CPMpy is a numpy-based library for conveniently modeling constraint programming problems in Python.

    Documentation in docs/index.rst
    as well as online at: https://cpmpy.readthedocs.io/

    Source code and bug reports at https://github.com/CPMpy/cpmpy

    The package constists of 4 modules:
    - `model`: a generic container for expressions (constraints and an objective), it can also search for an available solver and call it
    - `expressions`: all forms of expression objects that allow you to specify constraints and objectives over variables
    - `solvers`: CPMpy classes that translate a model into approriate calls of a solver's API
    - `transformations`: common methods for transforming expressions into other expressions, used by `solvers` modules to simplify/rewrite expressions
"""
# Tias Guns, 2019-2023

__version__ = "0.9.18"


from .expressions import *
from .model import Model
from .solvers.utils import SolverLookup
