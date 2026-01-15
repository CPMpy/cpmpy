# Adding a new solver

Any solver that has a Python interface can be added as a solver to CPMpy. See the bottom of this page for tips in case the/your solver does not have a Python interface yet.

To add your solver to CPMpy, you should copy [cpmpy/solvers/TEMPLATE.py](https://github.com/CPMpy/cpmpy/blob/master/cpmpy/solvers/TEMPLATE.py) directory, rename it to your solver name and start filling in the template. You can also look at how it is done for other solvers, they all follow the template.

Implementing the template consists of the following parts:

  * `supported()` where you check if the solver package is installed. Never include the solver python package at the top-level of the file, CPMpy has to work even if a user did not install your solver package.
  * `__init__()` where you initialize the underlying solver object
  * `solver_var()` where you create new solver variables and map them to CPMpy decision variables
  * `solve()` where you call the solver, get the status and runtime, and reverse-map the variable values after solving
  * `objective()` if your solver supports optimisation
  * `transform()` where you call the necessary transformations in `cpmpy.transformations` to transform CPMpy expressions to those that the solver supports
  * `__add__()` where you call transform and map the resulting CPMpy expressions, that the solver supports, to API function calls on the underlying solver
  * `solveAll()` optionally, if the solver natively supports solution enumeration

For your new solver to appear in CPMpy's [API documentation](./api/solvers.rst), add a `.rst` file in ``/docs/api/solvers`` (copy one of the other solvers' file and make the necessary changes) and add your solver to the *"List of submodules"* and the *"List of classes"* in the file ``/cpmpy/solvers/__init__.py``. And also to the overview table of solvers in ``docs/index.rst``. To test for incremental capabilities, use the ``examples/advanced/test_incremental_solving.py`` script.

To ease the installation process of your solver, be sure to add it to CPMpy's `setup.py`. Simply add an entry to `solver_dependencies`, mapping the cpmpy-native name for your solver (the same one used when calling `model.solve(solver=<SOLVER_NAME>)`) to the name of your pip-installable Python package and all its required dependencies.

## Transformations and posting constraints

CPMpy solver interfaces are *eager*, meaning that any CPMpy expression given to it (through `__add__()`) is immediately transformed (throught `transform()`) and then posted to the solver.

CPMpy is designed to separate *transforming* arbitrary CPMpy expressions to constraints the solver supports, from actually *posting* the supported constraints directly to the solver.

For example, a SAT solver only accepts clauses (disjunctions) as constraints. So, its `transform()` method has the challenge of mapping an arbitrary CPMpy expression to CPMpy 'or' expressions. This is exactly the task of a constraint modelling language like CPMpy, and we implement it through multiple solver-independent **transformation functions** in the `cpmpy/transformations/` directory that can achieve that and more. You hence only need to chain the right transformations in the solver's `transform()` method. It is best to look at a solver accepting a similar input, to see what transformations (and in what order) that one uses. 

The `__add__()` method will first call this `transform()`. This will return a list of CPMpy 'or' expression over decision variables. It then only has to iterate over those and call the solver its native API to create such clauses. All other constraints may not be directly supported by the solver, and can hence be rejected.

So for any solver you wish to add, chances are that most of the transformations you need are already implemented. Any solver can use any transformation in any order that the transformations allow. If you need additional transformations, or want to know how they work, read on.

## Stateless transformation functions

Because CPMpy's solver-interfaces transform and post constraints *eagerly*, they can be used *incremental*, meaning that you can add some constraints, call `solve()` add some more constraints and solve again. If the underlying solver is also incremental, it will reuse knowledge of the previous solve call to speed up this solve call.

The way that CPMpy succeeds to be an incremental modeling language, is by making all transformation functions *stateless*. Every transformation function is a python *function* that maps a (list of) CPMpy expressions to (a list of) equivalent CPMpy expressions. Transformations are not classes, they do not store state, they do not know (or care) what model a constraint belongs to. They take expressions as input and compute expressions as output. That means they can be called over and over again, and chained in any combination or order.

That also makes them modular, and any solver can use any combination of transformations that it needs. We continue to add and improve the transformations, and we are happy to discuss transformations you are missing, or variants of existing transformations that can be refined.

Most transformations do not need any state, they just do a bit of rewriting. Some transformations do, for example in the case of [Common Subexpression Elimination (CSE)](https://en.wikipedia.org/wiki/Common_subexpression_elimination). In that case, the solver interface (you who are reading this), should store a dictionary in your solver interface class, and pass that as (optional) argument to the transformation function. The transformation function will read and write to that dictionary as it needs, while still remaining stateless on its own. Each transformation function documents when it supports an optional state dictionary, see all available transformations in `cpmpy/transformations/`.


## What is a good Python interface for a solver?

A *light-weight, functional* API is what is most convenient from the CPMpy perspective, as well as in terms of setting up the Python-C++ bindings (or C, or whatever language the solver is written in).

With **functional** we mean that the API interface is for example a single class that has functions for adding variables, constraints and solve actions that it supports.

What we mean with **light-weight** is that it has none or few custom data-structures exposed at the Python level. That means that the arguments and return types of the API consist mostly of standard integers/strings/lists.

Here is fictional pseudo-code of such an API, which is heavily inspired on the OR-Tools CP-SAT interface:

```cpp
class SolverX {
    private Smth real_solver;

    // constructor
    void SolverX() {
        real_solver = ...; // internal solver object, not exported to Python
    }

    // managing variables
    str addBoolVar(str name); // returns unique variable ID (can also be a light-weight struct)
    str addIntVar(int lb, int ub, str name): // returns unique variable ID

    int getVarValue(str varID); // obtaining the value of a variable after solve

    // adding constraints
    void postAnd(vector<str> varIDs);
    void postAndImplied(str boolID, vector<str> varIDs); // bool implies and(vars)
    void postOr(vector<str> varIDs);
    void postOrImplied(str boolID, vector<str> varIDs);
    void postAllDifferent(vector<str> varIDs);
    void postSum(vector<str> varIds, str Operator, str varID);
    void postSum(vector<str> varIds, str Operator, int const);
    // I think OR-Tools actually creates a map (unique ID) for both variables and constants, so they can be used in the same expression
    void postWeightedSum(vector<str> varIds, vector<int> weights, str Operator, str varID);
    ...

    // adding objective
    void setObjective(str varID, bool is_minimize);
    void setObjectiveSum(vector<str> varID, bool is_minimize);
    void setObjectiveWeightedSum(vector<str> varID, vector<int> weights, bool is_minimize);
    ...

    // solving
    int solve(bool param1, int param2, str param3, ...); // return-value represents return state (opt, sat, unsat, error, ...)
    ...
}
```

If you have such a C++ API, then there exist automatic python packages that can make Python bindings, such as [CPPYY](https://cppyy.readthedocs.io/en/latest/) or [pybind11](https://pybind11.readthedocs.io/en/stable/).
For Rust, similar packages are available, such as [PyO3](https://pyo3.rs).

We have not done this ourselves yet, so get in touch to share your experience and advice!

## Testing your solver
The CPMpy package provides a large testsuite on which newly added solvers can be tested.
Note that for this testsuite to work, you need to add your solver to the `SolverLookup` utility.
This is done by adding an import statement in `/solvers/__init__.py` and adding an entry in the list of solvers in  `/solvers/utils.py`.

To run the testsuite on your solver, go to `/tests/test_constraints.py` and set `SOLVERNAMES` to the name of your solver. By running the file, every constraint allowed by the Flat Normal Form will be generated and posted to your solver interface.
As not every solver should support all possible constraints, you can exclude some using the `EXCLUDE_GLOBAL`, `EXCLUDE_OPERATORS` and `EXCLUDE_IMPL` dictionaries.
After posting the constraint, the answer of your solver is checked so you will both be able to monitor when your interface crashes or when a translation to the solver is incorrect.

## Tunable hyperparameters
CPMpy offers a tool for searching the best hyperparameter configuration for a given model on a solver (see [corresponding documentation](./solver_parameters.md)).
Solvers wanting to support this tool should add the following attributes to their interface: `tunable_params` and `default_params` (see [OR-Tools](https://github.com/CPMpy/cpmpy/blob/11ae35b22357ad9b8d6f47317df2c236c3ef5997/cpmpy/solvers/ortools.py#L473) for an example).


