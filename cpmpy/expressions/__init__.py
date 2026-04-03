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
    Count,
    Among,
    NValue,
    NValueExcept,
)
from .core import BoolVal
from .python_builtins import all, any, max, min, sum, abs

__all__ = [
    "Abs",
    "AllDifferent",
    "AllDifferentExcept0",
    "AllDifferentExceptN",
    "AllEqual",
    "AllEqualExceptN",
    "Among",
    "BoolVal",
    "BoolVar",
    "Circuit",
    "Count",
    "Cumulative",
    "CumulativeOptional",
    "Decreasing",
    "DecreasingStrict",
    "DirectConstraint",
    "Division",
    "Element",
    "GlobalCardinalityCount",
    "IfThenElse",
    "InDomain",
    "Increasing",
    "IncreasingStrict",
    "IntVar",
    "Inverse",
    "LexChainLess",
    "LexChainLessEq",
    "LexLess",
    "LexLessEq",
    "Maximum",
    "Minimum",
    "Modulo",
    "Multiplication",
    "NegativeTable",
    "NoOverlap",
    "NoOverlapOptional",
    "NValue",
    "NValueExcept",
    "Power",
    "Precedence",
    "Regular",
    "ShortTable",
    "Table",
    "Xor",
    "abs",
    "alldifferent",
    "allequal",
    "all",
    "any",
    "boolvar",
    "circuit",
    "cpm_array",
    "cparray",
    "intvar",
    "max",
    "min",
    "sum",
]
