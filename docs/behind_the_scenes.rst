Behind the scenes: CPMpy's pipeline
===================================

CPMpy conceptually has two key parts:

1. the 'language' that allows expressing constraint programming problems,
2. a mechanism to translate this language to the API of solvers.

Implementation wise, CPMpy has the following structure:

   - cpmpy/
      - model.py
      - expressions/
      - solvers/
      - transformations/

Everything related to the language is contained in the `cpmpy.expressions` module. The other modules support the translation and solving.


The language
------------
When you write a CPMpy model (constraints and an objective), you use Python's operators (`*,+,sum,-,~,|,&` etc) on CPMpy variables, as well as CPMpy functions and global constraints.

CPMpy uses Python's operator overloading to build expression trees (see expression.py). From CPMpy's point of view, a constraint programming problem is a list of expression trees (each one representing a constraint) and an expression tree for the objective function. You can write very complex nested expressions (e.g. `(a | b) == ((x + y > 5) -> (c & d))`), the language itself has few restrictions.

CPMpy only does minor modifications to the expressions when building the expression trees, e.g. it removes constants when chaining operators (e.g. `x + 0` :: `x`).

So the language offers acces to the high level expressions written by the user.


But solvers can't use this...

The mechanism
-------------
We have a number of staged transformations that the expression trees go through. These roughly correspond to different 'normal forms' as one would do in SAT, however, there are no 'normal forms' for constraint specifications as far as we are aware.

So far, we have the following 3 stages: ::

    CPMpy expression trees -> flatten -> solver-specific transf -> solver API
    |--------------------|    |-----|    |-----------------------------------|

As said, CPMpy expression trees allow arbitrary nesting, but only modeling languages (like MiniZinc and XCSP3) allow that. So if we want to use a solver API directly, we need to 'flatten' the nested expressions first.

Then, every solver has its own API, as well as some peculiarities (e.g. OR-Tools only supports implication/half-reification `->`, not standard double reification `==` (sometimes written as `<->`). These transformations are bundled into the solver-specific file in CPMpy.

Flattened 'normal form'
-----------------------

**This part needs updating, see cpmpy.transformations.flatten_model for the latest docs***

So that leaves the question, what is and is not allowed in this 'flattened' inbetween output?

Ideally, we can come up with a grammar that determines a formal normal form. By lack of that, here is a more informal description that is grammar-like. ::

   Var = IntVar | BoolVar | Num
   BoolVar = BoolVarImpl | NegBoolView | True | False

A variable is either an Integer decision variable, or a Boolean decision variable, or a numeric constant. For Boolean decision variables we have a special case: it is either an actual Boolean decision variable, or the negation of a Boolean decision variable (a negated 'view' on the variable), or the trivial True/False. ::

   BaseCons = ("name", [Var])

A 'base' constraint is simply a name, with a list of variables (no nested expressions). This includes global constraints, but also conjunction, disjunction, equality, etc.

To support linear constraints and reification (equating the truth-value of a constraint to a Boolean variable), we allow a few cases where a comparison operator can have a base constraint as its left-hand side.


### Special case 1, linear constraints: 

We first define a linear expression as follows (weighted linear sum): ::

   LinExpr = ("sum", ([Constant], [Var]))

This can be used in a linear constraint, or in the objective function: ::

   Obj = Var | LinExpr

# TODO: what about Max(), Min(), e.g. for makespan? Should be a standard operator?

A constraint that adds a comparison operator on a linear expression has two forms: ::

   LinConsRel = ("==", (LinExpr, Var)) | ("!=", (LinExpr, Var))

So (dis)equality can have a variable (or a constant) as its right-hand side. Inequality comparison operators will not: ::

   Op = ">" | ">=" | "<" | "<="
   LinConsIne = (Op, (LinExpr, Num))


### Special case 2, reification:

We first define the Boolean expressions that allow reification: ::

   BoolExpr = ("and", [Var]) | ("or", [Var]) | LinConsRel | LinConsIne

# TODO: some globals may also support reification... check and update here

Now, like linear constraints, in case of reification and implication, the left-hand side can be a simple Boolean expression with the right-hand side a Boolean variable: ::

   Reif = ("==", (BoolExpr, BoolVar))
   Impl = ("->", (BoolExpr, BoolVar))
