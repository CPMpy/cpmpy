"""
    All forms of expression objects that allow you to specify constraints and objectives over variables

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
from .globalconstraints import AllDifferent, AllEqual, Circuit, Table, Minimum, Maximum, Element
from .globalconstraints import alldifferent, allequal, circuit # Old, to be deprecated
from .python_builtins import all, any, max, min
