# Adding a new solver

Any solver that has a Python interface can be added as a solver to CPMpy. See lower on this page for tips in case your solver does not have a Python interface yet.

To add your solver to CPMpy, you should copy [cpmpy/solvers/TEMPLATE.py](https://github.com/CPMpy/cpmpy/blob/master/cpmpy/solvers/TEMPLATE.py), rename it to your solver name and start filling in the template. You can also look at how it is done for other solvers, they all follow the template.

Implementing the template consists of the following parts:

  * `version()` where you return the installed version of the solver's Python API, if its installed.
  * `supported()` where you check whether the solver is ready to use. Never include the solver python package at the top-level of the file, CPMpy has to work even if a user did not install your solver package. If needed, split this into helper checks such as `installed()`, `license_ok()`, or `executable_installed()`. For package version constraints, call ``cls._warn_outdated_dependencies()`` after a successful import; it checks the constraints of your extras key in ``setup.py`` (derived from the class name, ``CPM_mysolver`` -> ``"mysolver"``). That warns when the installed version is outside the declared range, without making ``supported()`` return ``False``, so users can still try an older optional solver dependency at their own risk.
  * `__init__()` and `native_model()` where you initialize and return the underlying solver object.
  * `solver_var()` where you create new solver variables and map them to CPMpy decision variables.
  * `solve()` where you call the solver, get the status and runtime, and reverse-map the variable values after solving.
  * `objective()` if your solver supports optimisation (optionally override `minimize`/`maximize`/`objective` with `Expression | FloatSum` type hints if your solver also supports :class:`~cpmpy.expressions.globalfunctions.FloatSum` objectives).
  * `supported_global_constraints` and `supported_reified_global_constraints` where you declare which integer functions and global constraints should reach the solver interface directly instead of being decomposed first.
  * `transform()` where you call the necessary transformations in `cpmpy.transformations` to transform CPMpy expressions to those that the solver supports.
  * `__add__()` where you call transform and map the resulting CPMpy expressions, that the solver supports, to API function calls on the underlying solver.

Now, to get your solver known and easy to use, you also have to register it in a number of places:

  * ``cpmpy/solvers/utils.py`` in the `SolverLookup.base_solvers()` function, the SOLVER_NAME you choose there is the one users will use when calling `model.solve(solver=<SOLVER_NAME>)`
  * ``README.md`` so the world knows its in CPMpy
  * ``docs/index.rst`` so the world knows its capabilities. To test for incremental capabilities, use the ``examples/advanced/test_incremental_solving.py`` script
  * ``docs/api/solvers/`` needs a `.rst` file for your solver, to appear in CPMpy's [API documentation](./api/solvers.rst) (copy one of the other solvers' file and make the necessary changes)
  * ``cpmpy/solvers/__init__.py`` in the *"List of classes"*, the imports and the all, so its easy to import from cpmpy.solvers
  * ``mypy.ini`` if your solver is not typed, you should set ignore_missing_imports for it here
  * ``setup.py``: add your package version constraints under your solver extras key (single source of truth, used by both install and the runtime version warnings); our policy is to warn about incompatible versions rather than hard-forbid them at runtime
  * ``.github/workflows/python-test.yml`` if the solver is free to use, then this will make the GitHub CI run the test-suite on every commit (highly recommended)
  * ``tests/test_solvers.py`` its not really required, but you can add one explicit test for your solver here, it will always run if the solver is installed
  * if you want your solver to be named in different places in the docs, check ``docs/solvers.md`` and ``docs/installation_instructions.rst`` for solvers mentioned there

Once the above works, consider connection optional extra solver features, if your solver supports them. These can also always be added in later commits.

  * `solution_hint()` for warm-starting the solver with a suggested variable assignment
  * `solve(solution_callback=..., display=...)` if the optimisation solver can return intermediate solutions during search
  * `solveAll()` if the solver natively supports solution enumeration
  * `solvernames()` and `solverversion()` if the interface exposes named subsolvers
  * `solve(assumptions=...)` and `get_core()` if the solver supports solving under assumptions and UNSAT core extraction
  * `mus_native()` if the solver has a native MUS/IIS extractor (used by `cpmpy.tools.explain)

## Transformations and posting constraints

CPMpy solver interfaces are *eager*, meaning that any CPMpy expression given to it (through `__add__()`) is immediately transformed (through `transform()`) and then posted to the solver.

CPMpy is designed to separate *transforming* arbitrary CPMpy expressions to constraints the solver supports, from actually *posting* the supported constraints directly to the solver.

For example, a SAT solver only accepts clauses (disjunctions) as constraints. So, its `transform()` method has the challenge of mapping an arbitrary CPMpy expression to CPMpy 'or' expressions. Transformations like these are exactly the task of a constraint modelling language like CPMpy, and we implement it through multiple solver-independent **transformation functions** in the `cpmpy/transformations/` directory that can achieve that and more. You hence only need to chain the right transformations in the solver's `transform()` method. It is best to look at a solver accepting a similar input, to see what transformations (and in what order) that one uses. 

The `__add__()` method will first call this `transform()`. This will return a list of CPMpy 'or' expression over decision variables. It then only has to iterate over those and call the solver its native API to create such clauses. All other constraints may not be directly supported by the solver, and can hence be rejected.

So for any solver you wish to add, chances are that most of the transformations you need are already implemented. A solver can choose the transformations it needs, but their order is not arbitrary: many transformations assume a particular input form, such as safe expressions, decomposed globals, flat normal form, or implication-only reification. Check the docstring of each transformation and look at solvers with a similar target language for a suitable chain. If you need additional transformations, or want to know how they work, read on.

## Stateless transformation functions

Because CPMpy's solver-interfaces transform and post constraints *eagerly*, they can be used *incremental*, meaning that you can add some constraints, call `solve()`, add some more constraints and solve again. If the underlying solver is also incremental, it will reuse knowledge of the previous solve call to speed up this solve call.

The way that CPMpy succeeds to be an incremental modeling language, is by making all transformation functions *stateless*. Every transformation function is a python *function* that maps a (list of) CPMpy expressions to (a list of) equivalent CPMpy expressions. Transformations are not classes, they do not store state, they do not know (or care) what model a constraint belongs to. They take expressions as input and compute expressions as output. Some transformations take solver-specific arguments, such as the set of supported globals. The output can depend on those arguments, but the transformation itself still does not keep hidden state between calls. That means they can be called over and over again. Do note that the order of transformations may matter as certain transformation functions may expect the input constraints to take a specific form, which is achieved by adding the necessary transformation before it.

Transformations are also modular, and any solver can use any combination of transformations that it needs. We continue to add and improve the transformations, and we are happy to discuss transformations you are missing, or variants of existing transformations that can be refined.

One exception to the stateless-ness of transformations is that we allow for [Common Subexpression Elimination (CSE)](https://en.wikipedia.org/wiki/Common_subexpression_elimination). In that case, the solver interface (you who are reading this), should store a dictionary in your solver interface class, and pass that as (optional) argument to the transformation function. The transformation function will read and write to that dictionary as it needs, while still remaining stateless on its own. Each transformation function documents when it supports an optional state dictionary, see all available transformations in `cpmpy/transformations/`.


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

To run the (extensive) testsuite on your solver, run:

```bash
python -m pytest tests/ --solver <SOLVER_NAME>
```

it will automatically test all of the allowed expressions through a constraint generator in `/tests/test_constraints.py`.
For a quicker feedback loop during development, you can run focused tests such as `tests/test_solverinterface.py` or `tests/test_solvers_solhint.py` with `--solver <SOLVER_NAME>` as well.
Using the transformation stack your solver should be able to handle all constraints and operators. However, during development there may be an exception to this rule.
You can exclude a global constraint or an operation using the `EXCLUDE_GLOBAL`, `EXCLUDE_OPERATORS` dictionaries respectively.
After posting the constraint, the answer of your solver is checked so you will both be able to monitor when your interface crashes or when a translation to the solver is incorrect.

Once your solver is passing the test suite, it is a good idea to check test coverage to see which lines of your solver code are never executed during the tests. Missing coverage may indicate missing tests, but it could also mean that code you intended to run was silently bypassed. For example, if you added native support for a constraint but a transformation is decomposing it before it reaches your solver, performance could suffer. You can generate an HTML coverage report in `htmlcov` using the [pytest-cov](https://pypi.org/project/pytest-cov/) plugin:

```bash
pip install pytest-cov
pytest --cov=cpmpy --cov-report=html -n auto
```

## Tunable hyperparameters
CPMpy offers a tool for searching the best hyperparameter configuration for a given model on a solver (see [corresponding documentation](./solver_parameters.md)).
Solvers wanting to support this tool should add the following classmethods to their interface: `tunable_params()` and `default_params()` (see [OR-Tools](https://github.com/CPMpy/cpmpy/blob/11ae35b22357ad9b8d6f47317df2c236c3ef5997/cpmpy/solvers/ortools.py#L473) for an example).


