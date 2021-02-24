CPMpy's pipeline
================

CPMpy has two key parts: 1) the 'language' that allows expressing constraint programming problems, 2) a mechanism to translate this language to the API of solvers.

The language
------------
When you write a CPMpy model (constraints and an objective), you use Python's operators (\*,+,sum,-,~,|,& etc) on CPMpy variables, as well as CPMpy functions and global constraints.

CPMpy uses' python's operator overloading capability to build an expressions trees. From CPMpy's point of view, a constraint programming problem is a list of expression trees (each one representing a constraint) and an expression tree for the objective function. You can write very complex nested expressions (e.g. (a | b) == ((x + y > 5) -> (c & d))), the language itself has few restrictions.

CPMpy only does minor modifications to the expressions when building the expression trees, e.g. it removes constants when chaining operators (e.g. x + 0 :: x)

So the language offers acces to the high level expressions written by the user.


But solvers can't use this...

The mechanism
-------------
We have a number of staged transformations that the expression trees go through. These roughly correspond to different 'normal forms' as one would do in SAT, however, there are no 'normal forms' for constraint specifications as far as we are aware.

So far, we have the following 3 stages:
CPMpy expression trees -> flatten -> solver-specific transf -> solver API
|--------------------|    |-----|    |-----------------------------------|

As said, CPMpy expression trees allow arbitrary nesting, but only modeling languages (like MiniZinc and XCSP3) allow that. So if we want to use a solver API directly, we need to 'flatten' the nested expressions first.

Then, every solver has its own API, as well as some peculiarities (e.g. or-tools only supports implication/half-reifiction `->`, not standard double reification `<->`. These transformations are bundled into the solver-specific file in CPMpy.

Flattened 'normal form'
-----------------------
So that leaves the question, what is and is not allowed in this 'flattened' inbetween output?

Ideally, we can come up with a grammar that determines a formal normal form. By lack of that, here is a more informal description that is grammar-like.

   Var = IntVar | BoolVar
   BoolVar = BoolVarImpl | NegBoolView

A variable is either an Integer decision variable, or a Boolean decision variable. For Boolean decision variables we have a special case: it is either an actual Boolean decision variable, or the negation of a Boolean decision variable (a negated 'view' on the variable).

   BaseCons = ("name", [Var])

A 'base' constraint is simply a name, with a list of variables (no nested expressions). This includes global constraints, but also conjunction, disjunction, equality, etc.

Special case 1, linear expressions:
   LinExpr = ("sum", ([Constant], [Var]))

A LinExpr can be used in a linear constraint, but also in the objective function.
   Obj = Var | LinExpr
# TODO: what about Max(), Min(), e.g. for makespan?

A linear constraint has a linear expression, an operator and a constant (# TODO, also another variable??)
   LinCons = ("op_name", (LinExpr, Constant))

Special case 2, reification:

Reification, like linear expressions, allow 1 level of nesting expressions on their left-hand side, more specifically:
   BoolExpr = ("and", [Var]) | ("or", [Var]) | LinCons
   Reif = ("==", (BoolExpr, BoolVar))
   Impl = ("->", (BoolExpr, BoolVar))
# TODO: do we also need to support global constraints, other expressions?
