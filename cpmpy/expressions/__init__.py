"""
Classes and functions that represent and create expressions (constraints and objectives)

==================
List of submodules
==================
.. autosummary::
    :nosignatures:

    variables
    core
    globalconstraints
    globalfunctions
    python_builtins
    utils


"""

# we only import methods/classes that are used for modelling
# others need to be imported by the developer explicitely
from .variables import boolvar, intvar, cpm_array
from .variables import BoolVar, IntVar, cparray # Old, to be deprecated
from .globalconstraints import (
    AllDifferent,
    AllDifferentExcept0,
    AllDifferentExceptN,
    AllEqual,
    AllEqualExceptN,
    Circuit,
    Inverse,
    Table,
    ShortTable,
    Xor,
    Cumulative,
    CumulativeOptional,
    IfThenElse,
    GlobalCardinalityCount,
    DirectConstraint,
    InDomain,
    Increasing,
    Decreasing,
    IncreasingStrict,
    DecreasingStrict,
    LexLess,
    LexLessEq,
    LexChainLess,
    LexChainLessEq,
    Precedence,
    NoOverlap,
    NoOverlapOptional,
    NegativeTable,
    Regular,
)
from .globalfunctions import (
    Minimum,
    Maximum,
    Abs,
    Multiplication,
    Division,
    Modulo,
    Power,
    Element,
    MultiDElement,
    Count,
    Among,
    NValue,
    NValueExcept,
)
from .core import BoolVal
from .python_builtins import all, any, max, min, sum, abs

__all__ = [
# Variables
    "boolvar",
    "intvar",
    "cpm_array",
# Variables (old, to be deprecated)
    "BoolVar",
    "IntVar",
    "cparray",
# Global functions
    "Minimum",
    "Maximum",
    "Abs",
    "Multiplication",
    "Division",
    "Modulo",
    "Power",
    "Element",
    "MultiDElement",
    "Count",
    "Among",
    "NValue",
    "NValueExcept",
# Global constraints
    "AllDifferent",
    "AllDifferentExcept0",
    "AllDifferentExceptN",
    "AllEqual",
    "AllEqualExceptN",
    "Among",
    "BoolVal",
    "Circuit",
    "Count",
    "Cumulative",
    "CumulativeOptional",
    "Decreasing",
    "DecreasingStrict",
    "DirectConstraint",
    "GlobalCardinalityCount",
    "IfThenElse",
    "InDomain",
    "Increasing",
    "IncreasingStrict",
    "Inverse",
    "LexChainLess",
    "LexChainLessEq",
    "LexLess",
    "LexLessEq",
    "NegativeTable",
    "NoOverlap",
    "NoOverlapOptional",
    "Precedence",
    "Regular",
    "ShortTable",
    "Table",
    "Xor",
# Python built-ins
    "abs",
    "all",
    "any",
    "max",
    "min",
    "sum",
]
