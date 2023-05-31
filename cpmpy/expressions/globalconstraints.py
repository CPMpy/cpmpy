#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## globalconstraints.py
##
"""
    Global constraints conveniently express non-primitive constraints.

    Using global constraints
    ------------------------

    Solvers can have specialised implementations for global constraints. CPMpy has GlobalConstraint
    expressions so that they can be passed to the solver as is when supported.

    If a solver does not support a global constraint (see solvers/) then it will be automatically
    decomposed by calling its `.decompose()` function.

    As a user you **should almost never subclass GlobalConstraint()** unless you know of a solver that
    supports that specific global constraint, and that you will update its solver interface to support it.

    For all other use cases, it sufficies to write your own helper function that immediately returns the
    decomposition, e.g.:

    .. code-block:: python

        def alldifferent_except0(args):
            return [ ((var1!= 0) & (var2 != 0)).implies(var1 != var2) for var1, var2 in all_pairs(args)]


    Numeric global constraints
    --------------------------

    CPMpy also implements __Numeric Global Constraints__. For these, the CPMpy GlobalConstraint does not
    exactly match what is implemented in the solver, but for good reason!!

    For example solvers may implement the global constraint `Minimum(iv1, iv2, iv3) == iv4` through an API
    call `addMinimumEquals([iv1,iv2,iv3], iv4)`.

    However, CPMpy also wishes to support the expressions `Minimum(iv1, iv2, iv3) > iv4` as well as
    `iv4 + Minimum(iv1, iv2, iv3)`. 

    Hence, the CPMpy global constraint only captures the `Minimum(iv1, iv2, iv3)` part, whose return type
    is numeric and can be used in any other CPMpy expression. Only at the time of transforming the CPMpy
    model to the solver API, will the expressions be decomposed and auxiliary variables introduced as needed
    such that the solver only receives `Minimum(iv1, iv2, iv3) == ivX` expressions.
    This is the burden of the CPMpy framework, not of the user who wants to express a problem formulation.


    Subclassing GlobalConstraint
    ----------------------------
    
    If you do wish to add a GlobalConstraint, because it is supported by solvers or because you will do
    advanced analysis and rewriting on it, then preferably define it with a standard decomposition, e.g.:

    .. code-block:: python

        class my_global(GlobalConstraint):
            def __init__(self, args):
                super().__init__("my_global", args)

            def decompose(self):
                return [self.args[0] != self.args[1]] # your decomposition

    If it is a __numeric global constraint__ meaning that its return type is numeric (see `Minimum` and `Element`)
    then set `is_bool=False` in the super() constructor and preferably implement `.value()` accordingly.


    Alternative decompositions
    --------------------------
    
    For advanced use cases where you want to use another decomposition than the standard decomposition
    of a GlobalConstraint expression, you can overwrite the 'decompose' function of the class, e.g.:

    .. code-block:: python

        def my_circuit_decomp(self):
            return [self.args[0] == 1] # does not actually enforce circuit
        circuit.decompose = my_circuit_decomp # attach it, no brackets!

        vars = intvar(1,9, shape=10)
        constr = circuit(vars)

        Model(constr).solve()

    The above will use 'my_circuit_decomp', if the solver does not
    natively support 'circuit'.

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        AllDifferent
        AllDifferentExcept0
        AllEqual
        Circuit
        Inverse
        Table
        Minimum
        Maximum
        Element
        Xor
        Cumulative
        Count
        GlobalCardinalityCount

"""
import copy
import warnings # for deprecation warning
import numpy as np
from ..exceptions import CPMpyException, IncompleteFunctionError, TypeError
from .core import Expression, Operator, Comparison
from .variables import boolvar, intvar, cpm_array, _NumVarImpl
from .utils import flatlist, all_pairs, argval, is_num, eval_comparison, is_any_list, is_boolexpr, get_bounds
from ..transformations.flatten_model import get_or_make_var

# Base class GlobalConstraint
class GlobalConstraint(Expression):
    """
        Abstract superclass of GlobalConstraints

        Like all expressions it has a `.name` and `.args` property.
        Overwrites the `.is_bool()` method. You can indicate
        in the constructor whether it has Boolean return type or not.
    """

    """
    Main functionalities:
    The GlobalConstraint class is an abstract superclass that represents a global constraint in Constraint Programming. 
    It has a name and a list of arguments, and can be either a Boolean or a numerical expression. It provides methods 
    for decomposition, negation, and checking whether it is a total function.

    Methods:
    - __init__(self, name, arg_list, is_bool=True): initializes a GlobalConstraint object with a name, a list of 
    arguments, and a Boolean flag indicating whether it is a Boolean expression or not.
    - is_bool(self): returns True if the GlobalConstraint is a Boolean expression, False otherwise.
    - decompose(self): returns a decomposition of the GlobalConstraint into smaller constraints.
    - get_bounds(self): returns the bounds of a Boolean GlobalConstraint.
    - decompose_negation(self): returns the negation of the GlobalConstraint.
    - is_total(self): returns True if the GlobalConstraint is a total function.

    Fields:
    - _is_bool: a Boolean flag indicating whether the GlobalConstraint is a Boolean expression or not.
    """

    # is_bool: whether this is normal constraint (True or False)
    #   not is_bool: it computes a numeric value (ex: Minimum, Element)
    def __init__(self, name, arg_list, is_bool=True):
        super().__init__(name, arg_list)
        self._is_bool = is_bool

    def is_bool(self):
        """ is it a Boolean (return type) Operator?
        """
        return self._is_bool

    def decompose(self):
        """
            Returns a decomposition into smaller constraints.

            The decomposition might create auxiliary variables
            and use other other global constraints as long as
            it does not create a circular dependency.
        """
        raise NotImplementedError("Decomposition for",self,"not available")

    def get_bounds(self):
        """
        Returns the bounds of a Boolean global constraint.
        Numerical global constraints should reimplement this.
        """
        return (0,1)

    def decompose_negation(self):
        from .python_builtins import all
        return [~all(self.decompose())]

    def is_total(self):
        """
            Returns whether the global constraint is a total function.
            If true, its value is defined for all arguments
        """
        return True

# Global Constraints (with Boolean return type)

def alldifferent(args):
    warnings.warn("Deprecated, use AllDifferent(v1,v2,...,vn) instead, will be removed in stable version", DeprecationWarning)
    return AllDifferent(*args) # unfold list as individual arguments
class AllDifferent(GlobalConstraint):
    """All arguments have a different (distinct) value
    """

    """
    Main functionalities:
    The AllDifferent class is a subclass of the GlobalConstraint class and represents the constraint that all arguments 
    have a different value. It is used to ensure that all variables in a given set of variables have distinct values.

    Methods:
    - __init__(self, *args): Initializes an instance of the AllDifferent class with a variable number of arguments.
    - decompose(self): Returns the decomposition of the constraint as a list of constraints that can be used to enforce 
    the constraint.
    - value(self): Returns True if all variables have distinct values, False otherwise.

    Fields:
    - name: The name of the constraint, which is "alldifferent".
    - args: The list of arguments to the constraint, which are the variables that need to have distinct values.
    """

    def __init__(self, *args):
        super().__init__("alldifferent", flatlist(args))

    def decompose(self):
        """Returns the decomposition
        """
        return [var1 != var2 for var1, var2 in all_pairs(self.args)]

    def value(self):
        return len(set(a.value() for a in self.args)) == len(self.args)


class AllDifferentExcept0(GlobalConstraint):
    """
    All nonzero arguments have a distinct value
    """

    """
    Main functionalities:
    The AllDifferentExcept0 class is a subclass of the GlobalConstraint class and represents a constraint that enforces that 
    all non-zero arguments have distinct values. It is used to model a problem where a set of variables must be assigned 
    distinct values, except for the value 0.

    Methods:
    - __init__(self, *args): Initializes an instance of the AllDifferentExcept0 class with a variable number of arguments.
    - decompose(self): Returns a list of constraints that enforce the AllDifferentExcept0 constraint.
    - value(self): Returns True if all non-zero arguments have distinct values, False otherwise.

    Fields:
    - Inherits the fields of the GlobalConstraint class, including name and args.
    """

    def __init__(self, *args):
        super().__init__("alldifferent_except0", flatlist(args))

    def decompose(self):
        return [((var1 != 0) & (var2 != 0)).implies(var1 != var2) for var1, var2 in all_pairs(self.args)]

    def value(self):
        vals = [a.value() for a in self.args if a.value() != 0]
        return len(set(vals)) == len(vals)


def allequal(args):
    warnings.warn("Deprecated, use AllEqual(v1,v2,...,vn) instead, will be removed in stable version", DeprecationWarning)
    return AllEqual(*args) # unfold list as individual arguments
class AllEqual(GlobalConstraint):
    """All arguments have the same value
    """

    """
    Main functionalities:
    The AllEqual class is a subclass of the GlobalConstraint class and represents a global constraint where all the 
    arguments have the same value. It is used to ensure that a set of variables or expressions take the same value in a 
    solution.

    Methods:
    - __init__(self, *args): Initializes an instance of the class with a list of arguments.
    - decompose(self): Returns a list of constraints that ensure that all the arguments have the same value.
    - value(self): Returns True if all the arguments have the same value, False otherwise.

    Fields:
    - Inherits the name and args fields from the GlobalConstraint class.
    """

    def __init__(self, *args):
        super().__init__("allequal", flatlist(args))

    def decompose(self):
        """Returns the decomposition
        """
        return [var1 == var2 for var1, var2 in all_pairs(self.args)]

    def value(self):
        return len(set(a.value() for a in self.args)) == 1


def circuit(args):
    warnings.warn("Deprecated, use Circuit(v1,v2,...,vn) instead, will be removed in stable version", DeprecationWarning)
    return Circuit(*args) # unfold list as individual arguments
class Circuit(GlobalConstraint):
    """The sequence of variables form a circuit, where x[i] = j means that j is the successor of i.
    """

    """
    Main functionalities:
    The Circuit class is a subclass of the GlobalConstraint class and represents a constraint that enforces a sequence 
    of variables to form a circuit, where x[i] = j means that j is the successor of i. The class provides a decomposition 
    method that returns a list of constraints that enforce the circuit constraint, and a value method that checks if the 
    current variable assignments satisfy the circuit constraint.

    Methods:
    - __init__(self, *args): Initializes a Circuit instance with a list of variables that represent the circuit.
    - decompose(self): Returns a list of constraints that enforce the circuit constraint.
    - value(self): Checks if the current variable assignments satisfy the circuit constraint.
    - decompose_negation(self): Returns the decomposition of the negation of the circuit constraint.

    Fields:
    - args: A list of variables that represent the circuit.
    """

    def __init__(self, *args):
        flatargs = flatlist(args)
        if any(is_boolexpr(arg) for arg in flatargs):
            raise TypeError("Circuit global constraint only takes arithmetic arguments: {}".format(flatargs))
        super().__init__("circuit", flatargs)
        if len(flatargs) < 2:
            raise CPMpyException('Circuit constraint must be given a minimum of 2 variables')

    def decompose(self):
        """
            Decomposition for Circuit

            Not sure where we got it from,
            MiniZinc has slightly different one:
            https://github.com/MiniZinc/libminizinc/blob/master/share/minizinc/std/fzn_circuit.mzn
        """
        succ = cpm_array(self.args)
        n = len(succ)
        order = intvar(0,n-1, shape=n)
        return [
            # different successors
            AllDifferent(succ),
            # different orders
            AllDifferent(order),
            # last one is '0'
            order[n-1] == 0,
            # loop: first one is successor of '0'
            order[0] == succ[0],
            # others: ith one is successor of i-1
        ] + [order[i] == succ[order[i-1]] for i in range(1,n)]


    def value(self):
        pathlen = 0
        idx = 0
        visited = set()
        arr = [argval(a) for a in self.args]
        while(idx not in visited):
            if idx == None:
                return False
            if not (0 <= idx < len(arr)):
                break
            visited.add(idx)
            pathlen += 1
            idx = arr[idx]

        return pathlen == len(self.args) and idx == 0

    def decompose_negation(self):
        '''
        returns the decomposition of the negation. We can not simply negate the decomposition
        because of the use of auxiliary variables in the decomposition

        should return something in negated normal form, since flatten_model.negated_normal() returns this
        '''
        from .python_builtins import all
        succ = cpm_array(self.args)
        n = len(succ)
        order = intvar(0, n - 1, shape=n)
        return [~all([AllDifferent(succ),
                # different orders
                AllDifferent(order)
                    ]),
                # not negating following constraints since they involve the auxiliary variables
                # loop: first one is successor of '0'
                order[0] == succ[0]
                ] + [order[i] == succ[order[i - 1]] for i in range(1, n)]


class Inverse(GlobalConstraint):
    """
       Inverse (aka channeling / assignment) constraint. 'fwd' and
       'rev' represent inverse functions; that is,

           fwd[i] == x  <==>  rev[x] == i

    """

    """
    Main functionalities:
    The Inverse class is a subclass of the GlobalConstraint class and represents the inverse (aka channeling/assignment) 
    constraint. It takes two arguments, fwd and rev, which represent inverse functions. The constraint ensures that if 
    fwd[i] == x, then rev[x] == i, and vice versa. It is used to establish a one-to-one correspondence between two sets 
    of variables.

    Methods:
    - __init__(self, fwd, rev): Initializes the Inverse constraint with the given forward and reverse functions.
    - decompose(self): Returns a decomposition of the constraint into smaller constraints. It creates auxiliary 
    variables and uses other global constraints to avoid circular dependencies.
    - value(self): Returns the value of the constraint. It checks if the constraint is satisfied for the current 
    variable assignments.
    - is_total(self): Returns True if the constraint is a total function, i.e., its value is defined for all arguments.

    Fields:
    - args: A list of the two arguments, fwd and rev, passed to the constructor.
    - name: The name of the constraint, which is "inverse".
    - _is_bool: A Boolean flag indicating whether the constraint has a Boolean return type. In this case, 
    it is always False.
    """

    def __init__(self, fwd, rev):
        flatargs = flatlist([fwd,rev])
        if any(is_boolexpr(arg) for arg in flatargs):
            raise TypeError("Only integer arguments allowed for global constraint Inverse: {}".format(flatargs))
        assert len(fwd) == len(rev)
        super().__init__("inverse", [fwd, rev])

    def decompose(self):
        from .python_builtins import all
        fwd, rev = self.args
        rev = cpm_array(rev)
        return [all(rev[x] == i for i, x in enumerate(fwd))]

    def value(self):
        fwd = [argval(a) for a in self.args[0]]
        rev = [argval(a) for a in self.args[1]]
        return all(rev[x] == i for i, x in enumerate(fwd))


"""
Main functionalities:
The Table class is a subclass of the GlobalConstraint class and represents a constraint that enforces that the values of 
the variables in the 'array' argument correspond to a row in the 'table' argument. It is used to model table constraints 
in constraint programming problems.

Methods:
- __init__(self, array, table): Constructor method that initializes the Table object with the 'array' and 'table' 
arguments.
- decompose(self): Method that returns a list of constraints that decompose the Table constraint into smaller 
constraints. It uses the 'any' and 'all' functions from the 'python_builtins' module to create the decomposition.
- value(self): Method that returns the value of the Table constraint. It checks if the current values of the variables 
in the 'array' argument correspond to a row in the 'table' argument.

Fields:
- args: A list of the 'array' and 'table' arguments passed to the constructor method.
"""
class Table(GlobalConstraint):
    """The values of the variables in 'array' correspond to a row in 'table'
    """
    def __init__(self, array, table):
        super().__init__("table", [array, table])

    def decompose(self):
        from .python_builtins import any, all
        arr, tab = self.args
        return [any(all(ai == ri for ai, ri in zip(arr, row)) for row in tab)]

    def value(self):
        arr, tab = self.args
        arrval = [argval(a) for a in arr]
        return arrval in tab


# syntax of the form 'if b then x == 9 else x == 0' is not supported
# a little helper:
class IfThenElse(GlobalConstraint):
    """
    Main functionalities:
    The IfThenElse class is a global constraint that represents an if-then-else statement. It takes in three boolean
    expressions as arguments: a condition, an expression to evaluate if the condition is true, and an expression to evaluate
    if the condition is false. The class can be used to model conditional statements in constraint programming problems.

    Methods:
    - __init__(self, condition, if_true, if_false): Initializes an IfThenElse object with a condition, an expression to
    evaluate if the condition is true, and an expression to evaluate if the condition is false.
    - value(self): Returns the value of the IfThenElse expression based on the current values of its arguments.
    - decompose(self): Returns a list of constraints that are equivalent to the IfThenElse expression.
    - __repr__(self): Returns a string representation of the IfThenElse expression.

    Fields:
    - name: The name of the IfThenElse global constraint.
    - args: A list of the three boolean expressions that make up the IfThenElse expression.
    - is_bool: A boolean value indicating whether the IfThenElse expression returns a boolean value or not.
    """

    def __init__(self, condition, if_true, if_false):
        if not is_boolexpr(condition) or not is_boolexpr(if_true) or not is_boolexpr(if_false):
            raise TypeError("only boolean expression allowed in IfThenElse")
        super().__init__("ite", [condition, if_true, if_false], is_bool=True)

    def value(self):
        condition, if_true, if_false = self.args
        condition_val = argval(condition)
        if argval(condition):
            return argval(if_true)
        else:
            return argval(if_false)

    def decompose(self):
        condition, if_true, if_false = self.args
        return [condition.implies(if_true), (~condition).implies(if_false)]

    def __repr__(self):
        condition, if_true, if_false = self.args
        return "If {} Then {} Else {}".format(condition, if_true, if_false)


class Minimum(GlobalConstraint):
    """
        Computes the minimum value of the arguments

        It is a 'functional' global constraint which implicitly returns a numeric variable
    """

    """
    Main functionalities:
    The Minimum class is a subclass of the GlobalConstraint class and represents a functional global constraint that 
    computes the minimum value of its arguments. It implicitly returns a numeric variable and can be used in comparisons.

    Methods:
    - __init__(self, arg_list): Initializes a new instance of the Minimum class with the given argument list. It calls the 
    constructor of the superclass and sets the is_bool field to False.
    - value(self): Returns the minimum value of the arguments. It first evaluates the arguments and returns None if any of 
    them is None. Otherwise, it returns the minimum value using the min() function.
    - decompose_comparison(self, cpm_op, cpm_rhs): Decomposes the global constraint if it is part of a comparison. It 
    creates a new integer variable and returns a list of constraints that enforce that the variable is the minimum of the 
    arguments and satisfies the comparison.
    - get_bounds(self): Returns the bounds of the numerical global constraint. It computes the bounds of each argument using
     the get_bounds() function and returns the minimum lower bound and the minimum upper bound.

    Fields:
    - args: A list of arguments.
    """
    def __init__(self, arg_list):
        super().__init__("min", flatlist(arg_list), is_bool=False)

    def value(self):
        argvals = [argval(a) for a in self.args]
        if any(val is None for val in argvals):
            return None
        else:
            return min(argvals)

    def decompose_comparison(self, cpm_op, cpm_rhs):
        """
        Decomposition if it's part of a comparison
        """
        from .python_builtins import any, all
        lb, ub = self.get_bounds()
        _min = intvar(lb, ub)
        return [any(x <= _min for x in self.args), all(x >= _min for x in self.args), eval_comparison(cpm_op, _min, cpm_rhs)]

    def get_bounds(self):
        """
        Returns the bounds of the (numerical) global constraint
        """
        bnds = [get_bounds(x) for x in self.args]
        return min(lb for lb,ub in bnds), min(ub for lb,ub in bnds)


class Maximum(GlobalConstraint):
    """
        Computes the maximum value of the arguments

        It is a 'functional' global constraint which implicitly returns a numeric variable
    """

    """
    Main functionalities:
    The Maximum class is a subclass of the GlobalConstraint class and represents the maximum value of a list of arguments. 
    It is a 'functional' global constraint which implicitly returns a numeric variable.

    Methods:
    - __init__(self, arg_list): Constructor method that initializes the Maximum object with a list of arguments.
    - value(self): Method that returns the maximum value of the arguments.
    - decompose_comparison(self, cpm_op, cpm_rhs): Method that returns a decomposition of the Maximum object if it's part of 
    a comparison.
    - get_bounds(self): Method that returns the bounds of the (numerical) global constraint.

    Fields:
    - args: A list of arguments.
    """

    def __init__(self, arg_list):
        super().__init__("max", flatlist(arg_list), is_bool=False)

    def value(self):
        argvals = [argval(a) for a in self.args]
        if any(val is None for val in argvals):
            return None
        else:
            return max(argvals)

    def decompose_comparison(self, cpm_op, cpm_rhs):
        """
        Decomposition if it's part of a comparison
        """
        from .python_builtins import any, all
        lb, ub = self.get_bounds()
        _max = intvar(lb, ub)
        return [any(x >= _max for x in self.args), all(x <= _max for x in self.args), eval_comparison(cpm_op, _max, cpm_rhs)]

    def get_bounds(self):
        """
        Returns the bounds of the (numerical) global constraint
        """
        bnds = [get_bounds(x) for x in self.args]
        return max(lb for lb,ub in bnds), max(ub for lb,ub in bnds)


def element(arg_list):
    warnings.warn("Deprecated, use Element(arr,idx) instead, will be removed in stable version", DeprecationWarning)
    assert (len(arg_list) == 2), "Element expression takes 2 arguments: Arr, Idx"
    return Element(arg_list[0], arg_list[1])
class Element(GlobalConstraint):
    """
        The 'Element' global constraint enforces that the result equals Arr[Idx]
        with 'Arr' an array of constants of variables (the first argument)
        and 'Idx' an integer decision variable, representing the index into the array.

        Solvers implement it as Arr[Idx] == Y, but CPMpy will automatically derive or create
        an appropriate Y. Hence, you can write expressions like Arr[Idx] + 3 <= Y

        Element is a CPMpy built-in global constraint, so the class implements a few more
        extra things for convenience (.value() and .__repr__()). It is also an example of
        a 'numeric' global constraint.
    """

    """
    Main functionalities:
    The Element class is a CPMpy built-in global constraint that enforces that the result equals Arr[Idx] with 'Arr' an 
    array of constants of variables (the first argument) and 'Idx' an integer decision variable, representing the index into 
    the array. It is a numeric global constraint that can be used in comparison relations and can be decomposed if used in a 
    comparison relation. It is also an example of a numeric global constraint.

    Methods:
    - __init__(self, arr, idx): Initializes the Element class with the array and index arguments.
    - value(self): Returns the value of the Element expression.
    - decompose_comparison(self, cpm_op, cpm_rhs): Decomposes the Element expression if used in a comparison relation.
    - is_total(self): Returns whether the global constraint is a total function.
    - __repr__(self): Returns a string representation of the Element expression.
    - get_bounds(self): Returns the bounds of the (numerical) global constraint.

    Fields:
    - args: The arguments of the Element expression.
    """

    def __init__(self, arr, idx):
        if is_boolexpr(idx):
            raise TypeError("index cannot be a boolean expression: {}".format(idx))
        super().__init__("element", [arr, idx], is_bool=False)

    def value(self):
        arr, idx = self.args
        idxval = argval(idx)
        if idxval is not None:
            if idxval >= 0 and idxval < len(arr):
                return argval(arr[idxval])
            raise IncompleteFunctionError(f"Index {idxval} out of range for array of length {len(arr)} while calculating value for expression {self}")
        return None # default

    def decompose_comparison(self, cpm_op, cpm_rhs):
        """
            `Element(arr,ix)` represents the array lookup itself (a numeric variable)
            It is not a constraint itself, so it can not have a decompose().
            However, when used in a comparison relation: Element(arr,idx) <CMP_OP> CMP_RHS
            it is a constraint, and that one can be decomposed.
            That is what this function does
            (for now only used in transformations/reification.py)
        """
        from .python_builtins import any

        arr, idx = self.args
        return [(idx == i).implies(eval_comparison(cpm_op, arr[i], cpm_rhs)) for i in range(len(arr))] + \
               [idx >= 0, idx < len(arr)]

    def is_total(self):
        arr, idx = self.args
        lb, ub = get_bounds(idx)
        return lb >= 0 & idx.ub < len(arr)

    def __repr__(self):
        return "{}[{}]".format(self.args[0], self.args[1])

    def get_bounds(self):
        """
        Returns the bounds of the (numerical) global constraint
        """
        arr, idx = self.args
        bnds = [get_bounds(x) for x in arr]
        return min(lb for lb,ub in bnds), max(ub for lb,ub in bnds)


class Xor(GlobalConstraint):
    """
        The 'xor' exclusive-or constraint
    """

    """
    Main functionalities:
    The Xor class is a subclass of the GlobalConstraint class and represents the exclusive-or constraint. It takes a list 
    of Boolean arguments and enforces that exactly one of them is true.

    Methods:
    - __init__(self, arg_list): initializes the Xor object with a list of Boolean arguments. It raises a TypeError if any 
    argument is not Boolean.
    - decompose(self): returns a decomposition of the Xor constraint into smaller constraints. It uses a logic decomposition
     that enforces that exactly one argument is true.
    - value(self): returns the value of the Xor constraint. It calculates the sum of the Boolean values of the arguments and
     returns True if the sum is odd, False otherwise.
    - __repr__(self): returns a string representation of the Xor constraint. It returns "arg1 xor arg2" if there are two 
    arguments, or "xor(arg1, arg2, ...)" if there are more than two.

    Fields:
    - args: a list of Expression objects representing the arguments of the Xor constraint.
    """

    def __init__(self, arg_list):
        flatargs = flatlist(arg_list)
        if not (all(is_boolexpr(arg) for arg in flatargs)):
            raise TypeError("Only Boolean arguments allowed in Xor global constraint: {}".format(flatargs))
        # convention for commutative binary operators:
        # swap if right is constant and left is not
        if len(arg_list) == 2 and is_num(arg_list[1]):
            arg_list[0], arg_list[1] = arg_list[1], arg_list[0]
            flatargs = arg_list
        super().__init__("xor", flatargs)

    def decompose(self):
        # there are multiple decompositions possible
        # sum(args) mod 2 == 1, for size 2: sum(args) == 1
        # since Xor is logical constraint, the default is a logic decomposition
        a0, a1 = self.args[:2]
        cons = (a0 | a1) & (~a0 | ~a1)  # one true and one false

        # for more than 2 variables, we cascade (decomposed) xors
        for arg in self.args[2:]:
            cons = (cons | arg) & (~cons | ~arg)
        return [cons]

    def value(self):
        return sum(argval(a) for a in self.args) % 2 == 1

    def __repr__(self):
        if len(self.args) == 2:
            return "{} xor {}".format(*self.args)
        return "xor({})".format(self.args)


class Cumulative(GlobalConstraint):
    """
        Global cumulative constraint. Used for resource aware scheduling.
        Ensures no overlap between tasks and never exceeding the capacity of the resource
        Supports both varying demand across tasks or equal demand for all jobs
    """

    """
    Main functionalities:
    The Cumulative class is a global constraint used for resource aware scheduling. It ensures that there is no overlap 
    between tasks and that the capacity of the resource is never exceeded. It supports both varying demand across tasks or 
    equal demand for all jobs.

    Methods:
    - __init__(self, start, duration, end, demand, capacity): initializes the Cumulative object with start, duration, end, 
    demand, and capacity arguments.
    - decompose(self): returns a time-resource decomposition of the Cumulative constraint.
    - value(self): returns True if the Cumulative constraint is satisfied, False otherwise.

    Fields:
    - start: a list of start times for each task.
    - duration: a list of durations for each task.
    - end: a list of end times for each task.
    - demand: a list of demands for each task, or a single demand value if all tasks have equal demand.
    - capacity: the capacity of the resource.
    """
    def __init__(self, start, duration, end, demand, capacity):
        assert is_any_list(start), "start should be a list"
        start = flatlist(start)
        assert is_any_list(duration), "duration should be a list"
        duration = flatlist(duration)
        for d in duration:
            if get_bounds(d)[0]<0:
                raise TypeError("durations should be non-negative")
        assert is_any_list(end), "end should be a list"
        end = flatlist(end)
        assert len(start) == len(duration) == len(end), "Lists should be equal length"

        if is_any_list(demand):
            demand = flatlist(demand)
            assert len(demand) == len(start), "Shape of demand should match start, duration and end"
            for d in demand:
                if is_boolexpr(d):
                    raise TypeError("demands must be non-boolean: {}".format(d))
        else:
            if is_boolexpr(demand):
                raise TypeError("demand must be non-boolean: {}".format(demand))
        flatargs = flatlist([start, duration, end, demand, capacity])
        if any(is_boolexpr(arg) for arg in flatargs):
            raise TypeError("All input lists should contain only arithmetic arguments for Cumulative constraints: {}".format(flatargs))

        super(Cumulative, self).__init__("cumulative",[start, duration, end, demand, capacity])

    def decompose(self):
        """
            Time-resource decomposition from:
            Schutt, Andreas, et al. "Why cumulative decomposition is not as bad as it sounds."
            International Conference on Principles and Practice of Constraint Programming. Springer, Berlin, Heidelberg, 2009.
        """
        from ..expressions.python_builtins import sum

        arr_args = (cpm_array(arg) if is_any_list(arg) else arg for arg in self.args)
        start, duration, end, demand, capacity = arr_args

        cons = []

        # set duration of tasks
        for t in range(len(start)):
            cons += [start[t] + duration[t] == end[t]]

        # demand doesn't exceed capacity
        lb, ub = min(s.lb for s in start), max(s.ub for s in end)
        for t in range(lb,ub+1):
            demand_at_t = 0
            for job in range(len(start)):
                if is_num(demand):
                    demand_at_t += demand * ((start[job] <= t) & (t < end[job]))
                else:
                    demand_at_t += demand[job] * ((start[job] <= t) & (t < end[job]))
            cons += [capacity >= demand_at_t]

        return cons

    def value(self):
        argvals = [np.array([argval(a) for a in arg]) if is_any_list(arg)
                   else argval(arg) for arg in self.args]

        if any(a is None for a in argvals):
            return None

        # start, dur, end are np arrays
        start, dur, end, demand, cap = argvals
        # start and end seperated by duration
        if not (start + dur == end).all():
            return False

        # demand doesn't exceed capacity
        lb, ub = min(start), max(end)
        for t in range(lb, ub+1):
            if cap < sum(demand * ((start <= t) & (t < end))):
                return False

        return True


class GlobalCardinalityCount(GlobalConstraint):
    """
        GlobalCardinalityCount(a,gcc): Collect the number of occurrences of each value 0..a.ub in gcc.
    The array gcc must have elements 0..ub (so of size ub+1).
        """

    """
    Main functionalities:
    The GlobalCardinalityCount class represents the global cardinality count constraint, which enforces that the number of 
    occurrences of each value in a given array is equal to the corresponding value in a given global cardinality count 
    array. The constraint is decomposed into a set of Count constraints, one for each value in the array, and the value of 
    the constraint is the conjunction of the values of these Count constraints.

    Methods:
    - __init__(self, a, gcc): Initializes a new instance of the GlobalCardinalityCount class with the given array and global
     cardinality count array.
    - decompose(self): Decomposes the constraint into a set of Count constraints, one for each value in the array.
    - value(self): Computes the value of the constraint as the conjunction of the values of the Count constraints.
    - get_bounds(self): Returns the bounds of the (numerical) global constraint.

    Fields:
    - args: The list of arguments to the constraint, consisting of the array and the global cardinality count array.
    """
    def __init__(self, a, gcc):
        flatargs = flatlist([a, gcc])
        if any(is_boolexpr(arg) for arg in flatargs):
            raise TypeError("Only numerical arguments allowed for gcc global constraint: {}".format(flatargs))
        ub = max([get_bounds(v)[1] for v in a])
        if not (len(gcc) == ub + 1):
            raise TypeError(f"GCC: length of gcc variables {len(gcc)} must be ub+1 {ub + 1}")
        super().__init__("gcc", [a,gcc])

    def decompose(self):
        a, gcc = self.args
        return [Count(a, i) == v for i, v in enumerate(gcc)]

    def value(self):
        from .python_builtins import all
        return all(self.decompose()).value()


class Count(GlobalConstraint):
    """
    The Count (numerical) global constraint represents the number of occurrences of val in arr
    """

    """
    Main functionalities:
    The Count class represents the Count (numerical) global constraint, which counts the number of occurrences of a 
    given value in an array.

    Methods:
    - __init__(self, arr, val): initializes a Count object with an array and a value to count
    - decompose_comparison(self, cmp_op, cmp_rhs): decomposes the Count constraint if it is part of a comparison
    - value(self): returns the number of occurrences of the value in the array
    - get_bounds(self): returns the lower and upper bounds of the Count constraint

    Fields:
    - args: a list of the array and value to count
    - name: the name of the Count constraint ("count")
    - is_bool: a boolean indicating whether the Count constraint returns a Boolean value (False) or not (True)
    """

    def __init__(self,arr,val):
        if is_any_list(val) or not is_any_list(arr):
            raise TypeError("count takes an array and a value as input, not: {} and {}".format(arr,val))
        super().__init__("count", [arr,val], is_bool=False)

    def decompose_comparison(self, cmp_op, cmp_rhs):
        """
        Count(arr,val) can only be decomposed if it's part of a comparison
        """
        arr, val = self.args
        return [eval_comparison(cmp_op, Operator('sum',[ai==val for ai in arr]), cmp_rhs)]

    def value(self):
        arr, val = self.args
        val = argval(val)
        return sum([argval(a) == val for a in arr])

    def get_bounds(self):
        """
        Returns the bounds of the (numerical) global constraint
        """
        arr, val = self.args
        return 0, len(arr)


class DirectConstraint(Expression):
    """
        A DirectConstraint will directly call a function of the underlying solver when added to a CPMpy solver

        It can not be reified, it is not flattened, it can not contain other CPMpy expressions than variables.
        When added to a CPMpy solver, it will literally just directly call a function on the underlying solver,
        replacing CPMpy variables by solver variables along the way.

        See the documentation of the solver (constructor) for details on how that solver handles them.

        If you want/need to use what the solver returns (e.g. an identifier for use in other constraints),
        then use `directvar()` instead, or access the solver object from the solver interface directly.
    """

    """
    Main functionalities:
    The DirectConstraint class represents a constraint that directly calls a function of the underlying solver when added to
    a CPMpy solver. It cannot be reified, flattened, or contain other CPMpy expressions than variables. When added to a 
    CPMpy solver, it will literally just directly call a function on the underlying solver, replacing CPMpy variables by 
    solver variables along the way. The class is used to interface with a specific solver and call its functions directly.

    Methods:
    - __init__(self, name, arguments, novar=None): initializes a DirectConstraint object with a name of the solver function 
    that you wish to call, a tuple of arguments to pass to the solver function with name 'name', and a list of indices 
    (offset 0) of arguments in `arguments` that contain no variables, that can be passed 'as is' without scanning for 
    variables.
    - is_bool(self): returns True, indicating that the return type of the DirectConstraint is Boolean.
    - callSolver(self, CPMpy_solver, Native_solver): calls the `directname`() function of the native solver, with stored 
    arguments replacing CPMpy variables with solver variables as needed. SolverInterfaces will call this function when this 
    constraint is added.

    Fields:
    - novar: a list of indices (offset 0) of arguments in `arguments` that contain no variables, that can be passed 'as is' 
    without scanning for variables.
    """

    def __init__(self, name, arguments, novar=None):
        """
            name: name of the solver function that you wish to call
            arguments: tuple of arguments to pass to the solver function with name 'name'
            novar: list of indices (offset 0) of arguments in `arguments` that contain no variables,
                   that can be passed 'as is' without scanning for variables
        """
        if not isinstance(arguments, tuple):
            arguments = (arguments,)  # force tuple
        super().__init__(name, arguments)
        self.novar = novar

    def is_bool(self):
        """ is it a Boolean (return type) Operator?
        """
        return True

    def callSolver(self, CPMpy_solver, Native_solver):
        """
            Call the `directname`() function of the native solver,
            with stored arguments replacing CPMpy variables with solver variables as needed.

            SolverInterfaces will call this function when this constraint is added.

        :param CPMpy_solver: a CPM_solver object, that has a `solver_vars()` function
        :param Native_solver: the python interface to some specific solver
        :return: the response of the solver when calling the function
        """
        # get the solver function, will raise an AttributeError if it does not exist
        solver_function = getattr(Native_solver, self.name)
        solver_args = copy.copy(self.args)
        for i in range(len(solver_args)):
            if self.novar is None or i not in self.novar:
                # it may contain variables, replace
                solver_args[i] = CPMpy_solver.solver_vars(solver_args[i])
        # len(native_args) should match nr of arguments of `native_function`
        return solver_function(*solver_args)

