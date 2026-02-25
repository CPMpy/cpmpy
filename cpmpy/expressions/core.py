#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## expressions.py
##
"""
    The :class:`~cpmpy.expressions.core.Expression` superclass and common subclasses :class:`~cpmpy.expressions.core.Comparison` and :class:`~cpmpy.expressions.core.Operator`.
    
    None of these objects should be directly created, they are automatically created through operator
    overloading on variables and expressions.

    Here is a list of standard python operators and what object (with what expr.name) it creates:

    Comparisons
    -----------
    ===================  ==========================
    Python Operator      CPMpy Object                     
    ===================  ==========================
    `x == y`             `Comparison("==", x, y)`         
    `x != y`             `Comparison("!=", x, y)`         
    `x < y`              `Comparison("<", x, y)`          
    `x <= y`             `Comparison("<=", x, y)`         
    `x > y`              `Comparison(">", x, y)`          
    `x >= y`             `Comparison(">=", x, y)`         
    ===================  ==========================

    Arithmetic Operators
    --------------------
    ===========================  ===============================================
    Python Operator              CPMpy Object                                  
    ===========================  ===============================================
    `-x`                         `Operator("-", [x])`                          
    `x + y`                      `Operator("sum", [x, y])`                     
    `sum([x,y,z])`               `Operator("sum", [x, y, z])`                  
    `sum([c0*x, c1*y, c2*z])`    `Operator("wsum", [[c0, c1, c2], [x, y, z]])` 
    `x - y`                      `Operator("sum", [x, -y])`                    
    `x * y`                      `globalfunctions.Multiplication(x, y)`
    `x // y`                     `globalfunctions.Division([x, y])` (integer division, rounding towards zero)
    `x % y`                      `globalfunctions.Modulo([x, y])` (remainder after integer division)
    `x ** y`                     `globalfunctions.Power([x, y])`
    ===========================  ===============================================

    
    Logical Operators
    -----------------
    ===================  =======================================================
    Python Operator      CPMpy Object  
    ===================  =======================================================                                        
    `x & y`              `Operator("and", [x, y])`                             
    `x | y`              `Operator("or", [x, y])`                              
    `~x`                 `Operator("not", [x])` or `NegBoolView(x)` if Boolean 
    `x ^ y`              `globalconstraints.Xor([x, y])`
    ===================  =======================================================

    Python has no built-in operator for `implication` that can be overloaded.
    CPMpy hence has a function :func:`~cpmpy.expressions.core.Expression.implies` that can be called:

    ===================  ======================
    Python Operator      CPMpy Object          
    ===================  ======================
    `x.implies(y)`       `Operator("->", [x,y])` 
    ===================  ======================

    Apart from operator overloading, expressions implement two important functions:

    - :func:`~cpmpy.expressions.core.Expression.is_bool`   
        which returns whether the return type of the expression is Boolean.
        If it does, the expression can be used as top-level constraint
        or in logical operators.

    - :func:`~cpmpy.expressions.core.Expression.value`    
        computes the value of this expression, by calling .value() on its
        subexpressions and doing the appropriate computation
        this is used to conveniently print variable values, objective values
        and any other expression value (e.g. during debugging).
    
    ===============
    List of classes
    ===============
    .. autosummary::
        :nosignatures:

        Expression
        Comparison
        Operator
"""
import copy
import warnings
from types import GeneratorType
import numpy as np
import cpmpy as cp

from .utils import is_int, is_num, is_any_list, flatlist, get_bounds, is_boolexpr, is_true_cst, is_false_cst, argvals, is_bool
from ..exceptions import IncompleteFunctionError, TypeError


class Expression(object):
    """
    An Expression represents a symbolic function with a `self.name` and `self.args` (arguments)

    Each Expression is considered to be a function whose value can be used
      in other expressions

    Expressions may implement:

    - :func:`~cpmpy.expressions.core.Expression.is_bool`:               whether its return type is Boolean
    - :func:`~cpmpy.expressions.core.Expression.value`:                 the value of the expression, default None
    - :func:`implies(x) <cpmpy.expressions.core.Expression.implies>`:   logical implication of this expression towards `x`
    - :func:`~cpmpy.expressions.core.Expression.__repr__`:              for pretty printing the expression
    - any ``__op__`` python operator overloading
    """

    def __init__(self, name, arg_list):
        self.name = name

        if isinstance(arg_list, (tuple, GeneratorType)):
            arg_list = list(arg_list)
        elif isinstance(arg_list, np.ndarray):
            # must flatten
            arg_list = arg_list.reshape(-1)
        for i in range(len(arg_list)):
            if isinstance(arg_list[i], np.ndarray):
                # must flatten
                arg_list[i] = arg_list[i].reshape(-1)

        assert (is_any_list(arg_list)), "_list_ of arguments required, even if of length one e.g. [arg]"
        self._args = arg_list


    @property
    def args(self):
        return self._args

    @args.setter
    def args(self, args):
        raise AttributeError("Cannot modify read-only attribute 'args', use 'update_args()'")

    def update_args(self, args):
        """ Allows in-place update of the expression's arguments.
            Resets all cached computations which depend on the expression tree.
        """
        self._args = args
        # Reset cached "_has_subexpr"
        if hasattr(self, "_has_subexpr"):
            del self._has_subexpr

    def set_description(self, txt, override_print=True, full_print=False):
        self.desc = txt
        self._override_print = override_print
        self._full_print = full_print

    def __str__(self):
        if not hasattr(self, "desc") or self._override_print is False:
            return self.__repr__()
        out = self.desc
        if self._full_print:
            out += " -- "+self.__repr__()
        return out


    def __repr__(self):
        strargs = []
        for arg in self.args:
            if isinstance(arg, np.ndarray):
                # flatten
                strarg = ",".join(map(str, arg.flat))
                strargs.append(f"[{strarg}]")
            else:
                strargs.append(f"{arg}")
        return "{}({})".format(self.name, ",".join(strargs))

    def __hash__(self):
        return hash(self.__repr__())

    def has_subexpr(self):
        """ Does it contains nested :class:`Expressions <cpmpy.expressions.core.Expression>` (anything other than a :class:`~cpmpy.expressions.variables._NumVarImpl` or a constant)?
            Is of importance when deciding whether certain transformations are needed
            along particular paths of the expression tree.
            Results are cached for future calls and reset when the expression changes
            (in-place argument update).
        """
        # return cached result
        if hasattr(self, '_has_subexpr'):
            return self._has_subexpr

        # Initialize stack with args
        stack = list(self.args)

        while stack:
            el = stack.pop()
            if isinstance(el, Expression):
                # only 3 types of expressions are leafs: _NumVarImpl, BoolVal or NDVarArray with no expressions inside.
                if isinstance(el, cp.variables.NDVarArray) and el.has_subexpr():
                    self._has_subexpr = True
                    return True
                elif not isinstance(el, (cp.variables._NumVarImpl, BoolVal)):
                    self._has_subexpr = True
                    return True
            elif is_any_list(el):
                # Add list elements to stack for processing
                stack.extend(el)

        # No subexpressions found
        self._has_subexpr = False
        return False

    def is_bool(self):
        """ is it a Boolean (return type) Operator?
            Default: yes
        """
        return True

    def value(self):
        return None # default

    def get_bounds(self):
        if self.is_bool():
            return 0, 1 #default for boolean expressions
        raise NotImplementedError(f"`get_bounds` is not implemented for type {self}")

    # keep for backwards compatibility
    def deepcopy(self, memodict={}):
        warnings.warn("Deprecated, use copy.deepcopy() instead, will be removed in stable version", DeprecationWarning)
        return copy.deepcopy(self, memodict)

    # implication constraint: self -> other
    # Python does not offer relevant syntax...
    # for double implication, use equivalence self == other
    def implies(self, other):
        # other constant
        if is_true_cst(other):
            return BoolVal(True)
        if is_false_cst(other):
            return ~self
        return Operator('->', [self, other])

    # Comparisons
    def __eq__(self, other):
        # BoolExpr == 1|true|0|false, common case, simply BoolExpr
        if self.is_bool() and is_num(other):
            if other is True or other == 1:
                return self
            if other is False or other == 0:
                return ~self
        return Comparison("==", self, other)

    def __ne__(self, other):
        return Comparison("!=", self, other)

    def __lt__(self, other):
        return Comparison("<", self, other)

    def __le__(self, other):
        return Comparison("<=", self, other)

    def __gt__(self, other):
        return Comparison(">", self, other)

    def __ge__(self, other):
        return Comparison(">=", self, other)

    # Boolean Operators
    # Implements bitwise operations & | ^ and ~ (and, or, xor, not)
    def __and__(self, other):
        # some simple constant removal
        if is_true_cst(other):
            return self
        # catch beginner mistake
        if is_num(other) and not is_bool(other):
            raise TypeError(f"{self}&{other} is not valid because {other} is a number, did you forget to put brackets? "
                            f"E.g. always write (x==2)&(y<5).")
        return Operator("and", [self, other])

    def __rand__(self, other):
        # some simple constant removal
        if is_true_cst(other):
            return self
        # catch beginner mistake
        if is_num(other) and not is_bool(other):
            raise TypeError(f"{other}&{self} is not valid because {other} is a number, "
                            f"did you forget to put brackets? E.g. always write (x==2)&(y<5).")
        return Operator("and", [other, self])

    def __or__(self, other):
        # some simple constant removal
        if is_false_cst(other):
            return self
        # catch beginner mistake
        if is_num(other) and not is_bool(other):
            raise TypeError(f"{self}|{other} is not valid because {other} is a number, "
                            f"did you forget to put brackets? E.g. always write (x==2)|(y<5).")
        return Operator("or", [self, other])

    def __ror__(self, other):
        # some simple constant removal
        if is_false_cst(other):
            return self
        # catch beginner mistake
        if is_num(other) and not is_bool(other):
            raise TypeError(f"{other}|{self} is not valid because {other} is a number, "
                            f"did you forget to put brackets? E.g. always write (x==2)|(y<5).")
        return Operator("or", [other, self])

    def __xor__(self, other):
        # some simple constant removal
        if is_true_cst(other):
            return ~self
        if is_false_cst(other):
            return self
        return cp.Xor([self, other])

    def __rxor__(self, other):
        # some simple constant removal
        if is_true_cst(other):
            return ~self
        if is_false_cst(other):
            return self
        return cp.Xor([other, self])

    # Mathematical Operators, including 'r'everse if it exists
    # Addition
    def __add__(self, other):
        if is_num(other) and other == 0:
            return self
        return Operator("sum", [self, other])

    def __radd__(self, other):
        if is_num(other) and other == 0:
            return self
        return Operator("sum", [other, self])

    # substraction
    def __sub__(self, other):
        # if is_num(other) and other == 0:
        #     return self
        # return Operator("sub", [self, other])
        return self.__add__(-other)

    def __rsub__(self, other):
        # if is_num(other) and other == 0:
        #     return -self
        # return Operator("sub", [other, self])
        return (-self).__radd__(other)
    
    # multiplication: use GlobalFunction Multiplication so it can be decomposed (e.g. to linear)
    def __mul__(self, other):
        if is_num(other) and other == 1:
            return self
        return cp.Multiplication(self, other)

    def __rmul__(self, other):
        if is_num(other) and other == 1:
            return self
        return cp.Multiplication(other, self)

    # matrix multipliciation TODO?
    #object.__matmul__(self, other)

    # other mathematical ones
    def __truediv__(self, other):
        warnings.warn("We only support floordivision, use // in stead of /", SyntaxWarning)
        return self.__floordiv__(other)

    def __rtruediv__(self, other):
        warnings.warn("We only support floordivision, use // in stead of /", SyntaxWarning)
        return self.__rfloordiv__(other)

    def __floordiv__(self, other):
        if is_num(other) and other == 1:
            return self
        return cp.Division(self, other)

    def __rfloordiv__(self, other):
        return cp.Division(other, self)

    def __mod__(self, other):
        return cp.Modulo(self, other)

    def __rmod__(self, other):
        return cp.Modulo(other, self)

    def __pow__(self, other, modulo=None):
        assert (modulo is None), "Power operator: modulo not supported"
        if is_num(other) and other == 1:
            return self
        return cp.Power(self, other)

    def __rpow__(self, other, modulo=None):
        assert (modulo is None), "Power operator: modulo not supported"
        return cp.Power(other, self)

    # Not implemented: (yet?)
    #object.__divmod__(self, other)

    # unary mathematical operators
    def __neg__(self):
        if self.name == 'wsum':
            # negate the constant weights
            return Operator(self.name, [[-a for a in self.args[0]], self.args[1]])
        return Operator("-", [self])

    def __pos__(self):
        return self

    def __abs__(self):
        return cp.Abs(self)

    def __invert__(self):
        if not (is_boolexpr(self)):
            raise TypeError("Not operator is only allowed on boolean expressions: {0}".format(self))
        return Operator("not", [self])

    def __bool__(self):
        raise ValueError(f"__bool__ should not be called on a CPMPy expression {self} as it will always return True\n"
                         "Do not use an expression as argument in an `if` statement and use cpmpy.any, cpmpy.max instead of python builtins\n"
                         "If you think this is an error, please report on github")

class BoolVal(Expression):
    """
        Wrapper for python or numpy BoolVals
    """

    def __init__(self, arg):
        assert is_true_cst(arg) or is_false_cst(arg), f"BoolVal must be initialized with a boolean constant, got {arg} of type {type(arg)}"
        super(BoolVal, self).__init__("boolval", [bool(arg)])

    def value(self):
        return self.args[0]

    def __invert__(self):
        return BoolVal(not self.args[0])

    def __bool__(self):
        """Called to implement truth value testing and the built-in operation bool(), return stored value"""
        return self.args[0]

    def __int__(self):
        """Called to implement conversion to numerical"""
        return int(self.args[0])

    def get_bounds(self):
        v = int(self.args[0])
        return (v,v)

    def __and__(self, other):
        if is_bool(other): # Boolean constant
            return BoolVal(self.args[0] and other)
        elif isinstance(other, Expression) and other.is_bool():
            if self.args[0]:
                return other
            else:
                return BoolVal(False)
        raise ValueError(f"{self}&{other} is not valid. Expected Boolean constant or Boolean Expression, but got {other} of type {type(other)}.")
        
    
    def __rand__(self, other):
        if is_bool(other): # Boolean constant
            return BoolVal(self.args[0] and other)
        elif isinstance(other, Expression) and other.is_bool():
            if self.args[0]:
                return other
            else:
                return BoolVal(False)
        raise ValueError(f"{self}&{other} is not valid. Expected Boolean constant or Boolean Expression, but got {other} of type {type(other)}.")

    
    def __or__(self, other):
        if is_bool(other): # Boolean constant
            return BoolVal(self.args[0] or other)
        elif isinstance(other, Expression) and other.is_bool():
            if not self.args[0]:
                return other
            else:
                return BoolVal(True)
        raise ValueError(f"{self}|{other} is not valid. Expected Boolean constant or Boolean Expression, but got {other} of type {type(other)}.")
        
        
    def __ror__(self, other):
        if is_bool(other): # Boolean constant
            return BoolVal(self.args[0] or other)
        elif isinstance(other, Expression) and other.is_bool():
            if not self.args[0]:
                return other
            else:
                return BoolVal(True)
        raise ValueError(f"{self}|{other} is not valid. Expected Boolean constant or Boolean Expression, but got {other} of type {type(other)}.")
        
    def __xor__(self, other):
        if is_bool(other): # Boolean constant
            return BoolVal(self.args[0] ^ other)
        elif isinstance(other, Expression) and other.is_bool():
            if self.args[0]:
                return ~other
            else:
                return other
        raise ValueError(f"{self}^^{other} is not valid. Expected Boolean constant or Boolean Expression, but got {other} of type {type(other)}.")
    
    
    def __rxor__(self, other):
        if is_bool(other): # Boolean constant
            return BoolVal(self.args[0] ^ other)
        elif isinstance(other, Expression) and other.is_bool():
            if self.args[0]:
                return ~other
            else:
                return other
        raise ValueError(f"{self}^^{other} is not valid. Expected Boolean constant or Boolean Expression, but got {other} of type {type(other)}.")
    

    def has_subexpr(self) -> bool:
        """ Does it contains nested Expressions (anything other than a _NumVarImpl or a constant)?
            Is of importance when deciding whether certain transformations are needed
            along particular paths of the expression tree.
        """
        return False # BoolVal is a wrapper for a python or numpy constant boolean.

    def implies(self, other):
        if self.args[0]:
            return other
        else:
            return other == other  # Always true, but keep variables in the model


class Comparison(Expression):
    """Represents a comparison between two sub-expressions
    """
    allowed = {'==', '!=', '<=', '<', '>=', '>'}

    def __init__(self, name, left, right):
        assert (name in Comparison.allowed), f"Symbol {name} not allowed"
        super().__init__(name, [left, right])

    def __repr__(self):
        if all(isinstance(x, Expression) for x in self.args):
            return "({}) {} ({})".format(self.args[0], self.name, self.args[1]) 
        # if not: prettier printing without braces
        return "{} {} {}".format(self.args[0], self.name, self.args[1]) 
    
    def __bool__(self):
        # will be called when comparing elements in a container, but always with `==`
        if self.name == "==":
            return repr(self.args[0]) == repr(self.args[1])
        super().__bool__() # default to exception

    # return the value of the expression
    # optional, default: None
    def value(self):
        arg_vals = argvals(self.args)

        if any(a is None for a in arg_vals): return None
        if   self.name == "==": return arg_vals[0] == arg_vals[1]
        elif self.name == "!=": return arg_vals[0] != arg_vals[1]
        elif self.name == "<":  return arg_vals[0] < arg_vals[1]
        elif self.name == "<=": return arg_vals[0] <= arg_vals[1]
        elif self.name == ">":  return arg_vals[0] > arg_vals[1]
        elif self.name == ">=": return arg_vals[0] >= arg_vals[1]
        return None # default


class Operator(Expression):
    """
    All kinds of mathematical and logical operators on expressions

    Convention for 2-ary operators: if one of the two is a constant,
    it is stored first (as expr[0]), this eases weighted sum detection
    """
    allowed = {
        #name: (arity, is_bool)       arity 0 = n-ary, min 2
        'and': (0, True),
        'or':  (0, True),
        '->':  (2, True),
        'not': (1, True),
        'sum': (0, False),
        'wsum': (2, False),
        'sub': (2, False), # x - y
        '-':   (1, False), # -x
    }
    printmap = {'sum': '+', 'sub': '-'}

    def __init__(self, name, arg_list):
        # sanity checks
        assert (name in Operator.allowed), "Operator {} not allowed".format(name)
        arity, is_bool_op = Operator.allowed[name]
        if is_bool_op:
            #only boolean arguments allowed
            for arg in arg_list:
                if not is_boolexpr(arg):
                    raise TypeError("{}-operator only accepts boolean arguments, not {}".format(name,arg))
        if arity == 0:
            arg_list = flatlist(arg_list)
            assert (len(arg_list) >= 1), "Operator: n-ary operators require at least one argument"
        else:
            assert (len(arg_list) == arity), "Operator: {}, number of arguments must be {}".format(name, arity)

        # automatic weighted sum (wsum) creation:
        # if all args are an expression (not a constant)
        #    and one of the args is a wsum,
        #                    or a product of a constant and an expression,
        # then create a wsum of weights,expressions over all
        if name == 'sum' and \
                all(not is_num(a) for a in arg_list) and \
                any(_wsum_should(a) for a in arg_list):
            we = [_wsum_make(a) for a in arg_list]
            w = [wi for w, _ in we for wi in w]
            e = [ei for _, e in we for ei in e]
            name = 'wsum'
            arg_list = [w, e]

        # we have the requirement that weighted sums are [weights, expressions]
        if name == 'wsum':
            assert all(is_num(a) for a in arg_list[0]), "wsum: arg0 has to be all constants but is: "+str(arg_list[0])
            weights = []
            for w in arg_list[0]:
                if is_int(w):
                    weights.append(int(w)) # bool or int, simplifies things later on
                else:
                    weights.append(w) # can be float
            arg_list = (weights, arg_list[1])

        # small cleanup: nested n-ary operators are merged into the toplevel
        # (this is actually against our design principle of creating
        #  expressions the way the user wrote them)
        if arity == 0:
            i = 0 # length can change
            while i < len(arg_list):
                if isinstance(arg_list[i], Operator) and arg_list[i].name == name:
                    # merge args in at this position
                    l = len(arg_list[i].args)
                    arg_list[i:i+1] = arg_list[i].args
                    i += l
                i += 1


        super().__init__(name, arg_list)

    def is_bool(self):
        """ is it a Boolean (return type) Operator?
        """
        return Operator.allowed[self.name][1]

    def __repr__(self):
        printname = self.name
        if printname in Operator.printmap:
            printname = Operator.printmap[printname]

        # special cases
        if self.name == '-': # unary -
            return "-({})".format(self.args[0])

        # weighted sum
        if self.name == 'wsum':
            return f"sum({self.args[0]} * {self.args[1]})"

        # infix printing of two arguments
        if len(self.args) == 2:
            # bracketed printing of non-constants
            def wrap_bracket(arg):
                if isinstance(arg, Expression):
                    return f"({arg})"
                return arg
            return "{} {} {}".format(wrap_bracket(self.args[0]),
                                     printname,
                                     wrap_bracket(self.args[1]))

        return "{}({})".format(self.name, self.args)

    def value(self):

        if self.name == "wsum":
            # wsum: arg0 is list of constants, no .value() use as is
            arg_vals = [self.args[0], argvals(self.args[1])]
        else:
            arg_vals = argvals(self.args)


        if any(a is None for a in arg_vals): return None
        # non-boolean
        elif self.name == "sum": return sum(arg_vals)
        elif self.name == "wsum":
            val = np.dot(arg_vals[0], arg_vals[1]).item()
            if round(val) == val: # it is an integer
                return int(val)
            return val # can be a float
        elif self.name == "sub": return arg_vals[0] - arg_vals[1]
        elif self.name == "-":   return -arg_vals[0]

        # boolean
        elif self.name == "and": return all(arg_vals)
        elif self.name == "or" : return any(arg_vals)
        elif self.name == "->": return (not arg_vals[0]) or arg_vals[1]
        elif self.name == "not": return not arg_vals[0]

        return None # default

    def get_bounds(self):
        """
        Returns an estimate of lower and upper bound of the expression.
        These bounds are safe: all possible values for the expression agree with the bounds.
        These bounds are not tight: it may be possible that the bound itself is not a possible value for the expression.
        """
        if self.is_bool():
            return 0, 1 #boolean
        elif self.name == 'sum':
            lbs, ubs = get_bounds(self.args)
            lowerbound, upperbound = sum(lbs), sum(ubs)
        elif self.name == 'wsum':
            weights, vars = self.args
            bounds = []
            lowerbound, upperbound = 0,0
            #this may seem like too many lines, but avoiding np.sum avoids overflowing things at int32 bounds
            for w, (lb, ub) in zip(weights, [get_bounds(arg) for arg in vars]):
                x,y = int(w) * lb, int(w) * ub
                if x <= y: # x is the lb of this arg
                    lowerbound += x
                    upperbound += y
                else:
                    lowerbound += y
                    upperbound += x

        elif self.name == 'sub':
            lb1, ub1 = get_bounds(self.args[0])
            lb2, ub2 = get_bounds(self.args[1])
            lowerbound, upperbound = lb1-ub2, ub1-lb2

        elif self.name == '-':
            lb1, ub1 = get_bounds(self.args[0])
            lowerbound, upperbound = -ub1, -lb1

        if lowerbound == None:
            raise ValueError(f"Bound requested for unknown expression {self}, please report bug on github")
        if lowerbound > upperbound:
            #overflow happened
            raise OverflowError(f'Overflow when calculating bounds, your expression exceeds integer bounds: {self}')
        return lowerbound, upperbound


def _wsum_should(arg):
    """ Internal helper: should the arg be in a wsum instead of sum

    True if the arg is already a wsum,
    or if it is a Multiplication with is_lhs_num
    (negation '-' does not mean it SHOULD be a wsum, because then
     all substractions are transformed into less readable wsums)
    """
    name = getattr(arg, 'name', None)
    return name == 'wsum' or (name == 'mul' and arg.is_lhs_num)

def _wsum_make(arg):
    """ Internal helper: prep the arg for wsum

    returns ([weights], [expressions]) where 'weights' are constants.
    Handles Operator (wsum, sum, -) and Multiplication (name 'mul', constant lhs when is_lhs_num).
    """
    name = getattr(arg, 'name', None)
    if name == 'wsum':
        return arg.args
    elif name == 'sum':
        return [1]*len(arg.args), arg.args
    elif name == 'mul':
        if arg.is_lhs_num:
            return [arg.args[0]], [arg.args[1]]
    elif name == '-':
        return [-1], [arg.args[0]]
    # default
    return [1], [arg]
