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
from .globalconstraints import AllDifferent, AllDifferentExcept0, AllDifferentExceptN, AllEqual, AllEqualExceptN, Circuit, Inverse, Table, ShortTable, Xor, Cumulative, \
    IfThenElse, GlobalCardinalityCount, DirectConstraint, InDomain, Increasing, Decreasing, IncreasingStrict, DecreasingStrict, \
    LexLess, LexLessEq, LexChainLess, LexChainLessEq, Precedence, NoOverlap, \
    NegativeTable, Regular
from .globalconstraints import alldifferent, allequal, circuit # Old, to be deprecated
from .globalfunctions import Minimum, Maximum, Abs, Multiplication, Division, Modulo, Power, Element, Count, Among, NValue, NValueExcept
from .core import BoolVal
from .python_builtins import all, any, max, min, sum, abs
