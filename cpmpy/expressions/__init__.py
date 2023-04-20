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
        python_builtins
        utils


"""

# we only import methods/classes that are used for modelling
# others need to be imported by the developer explicitely
from .variables import boolvar, intvar, cpm_array
from .variables import BoolVar, IntVar, cparray # Old, to be deprecated
from .globalconstraints import AllDifferent, AllDifferentExcept0, AllEqual, Circuit, Inverse, Table, Minimum, Maximum, Element, Xor, Cumulative, IfThenElse, Count, GlobalCardinalityCount
from .globalconstraints import alldifferent, allequal, circuit # Old, to be deprecated
from .core import BoolVal
from .python_builtins import all, any, max, min, sum
