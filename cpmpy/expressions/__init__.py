"""
    All forms of expression objects that allow you to specify constraints and objectives over variables

    Contains the following submodules:
    - expressions: the `Expression` superclass and common subclasses. None of these objects need to be directly created, they are created through operator overloading on variables, or through helper functions (global constraints)
    - variables: integer and boolean variables as n-dimensional numpy objects
    - python_builtins: overwrites a number of python built-ins, so that they work over variables as expected
    - globalconstraints: functions for expressing constraints that can not be expressed through operator overloading, or that have special handling routines in solvers
    - utils: internal utilities for expression handling
"""

# we only import methods/classes that are used for modelling
# others need to be imported by the developer explicitely
from .variables import BoolVar, IntVar, cparray
from .globalconstraints import alldifferent, allequal, circuit, Table, Minimum, Maximum, Element
from .python_builtins import all, any, max, min
