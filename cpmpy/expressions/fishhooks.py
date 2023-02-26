#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## fishhooks.py
##
"""
    Hooks (and overrides) a number of functions to built-in python types.

    =================
    List of functions
    =================
    .. autosummary::
        :nosignatures:

        bool:
        __and__
        __or__
        implies

        int:
        is_bool
        __and__
        __or__
        implies
"""


from .variables import _BoolVarImpl
from .core import Expression, Operator
from fishhook import hook, orig


@hook(bool, int)
def __and__(self, other):
    if isinstance(other, Expression):
        return other.__rand__(self)
    return orig(self, other)

@hook(bool, int)
def __or__(self, other):
    if isinstance(other, Expression):
        return other.__ror__(self)
    return orig(self, other)

@hook(bool, int)
def implies(self, other):
    if not (isinstance(other,Expression)):
        return orig(self, other)
    else:

        assert self.is_bool(), f"Logical implication involving an integer ({self}) is not allowed"

        if isinstance(self,int):
            self = (self == 1)  # convert to True or False

        assert isinstance(other,_BoolVarImpl), f"Logical implication involving an IntVar ({other}) is not allowed"

        return Operator("->", [self, other])

@hook(int)
def is_bool(self):
    if self > 1 or self < 0:
        return False
    else:
        return True
