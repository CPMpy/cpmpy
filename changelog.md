# Change log

## 0.9.14
Hotfix release

Our builtin-overwrites 'any' and 'all' returned BoolVal's, and they did not yet have a `__bool__()` property so they would be correctly evaluated when used in 'if' functions and other standard python contexts. This can easily break user code that uses `any` or `all` when doing 'from cpmpy import \*'.

Unfortunately we merged the BoolVal branch with the above behaviour, even though we had a version that fixed it, but this was not yet pushed to the branch.

* This hotfix release fixes it so that `any` and `all` return standard Booleans again (and BoolVal has a `__bool__()` also).

* We also include a small fix to make the Inverse global constraint decomposition work for non-variable arrays too.


## 0.9.13
Solid progress release.

To make more clear how well-supported each of the solvers are, we introduced a tiered classification:

* Tier 1 solvers: passes all internal tests, passes our bigtest suit, will be fuzztested in the near future
    - "ortools" the OR-Tools CP-SAT solver
    - "pysat" the PySAT library and its many SAT solvers ("pysat:glucose4", "pysat:lingeling", etc)

* Tier 2 solvers: passes all internal tests, might fail on edge cases in bigtest
    - "minizinc" the MiniZinc modeling system and its many solvers ("minizinc:gecode", "minizinc:chuffed", etc)
    - "z3" the SMT solver and theorem prover
    - "gurobi" the MIP solver
    - "PySDD" a Boolean knowledge compiler

* Tier 3 solvers: they are work in progress and live in a pull request
    - "gcs" the Glasgow Constraint Solver
    - "exact" the Exact pseudo-boolean solver

We hope to upgrade many of these solvers to higher tiers, as well as adding new ones. Reach out on github if you want to help out.

New above the hood:
* added 'DirectConstraint', a generic way to post solver-specific constraints which CPMpy does not implement, with multiple examples such as ortools' automaton, circuit, multicircuit
* added 'Count' global constraint with decomposition
* added 'GlobalCardinalityCount' (GCC) global constraint with decomposition
* added 'Inverse' global constraint with with decomposition
* a Boolean 'IfThenElse' global constraint with decomposition c->if_true & (~c)->if_false

New under the hood:
* a BoolVal() expression object for constants True/False, better handling/cleaning of Bool constants as a result
* added a highly efficient 'toplevel_list' transformation that all solvers call to get a list of CPMpy expressions, simplifies what to expect as input for transformations
* 'decompose_globals' is now a transformation that decomposes the unsupported globals, it also does it best to properly handle 'numeric' globals, reified globals and negated globals

Changed:
* added missing decomposition for 'Table' global constraint
* highly optimized the 'get_variables' transformation
* pushed bounds computation into the expressions, more robust and extensible
* removed custom deepcopy() for Python's better built-in one
* slightly better handling of incomplete (partial) functions, e.g. in bounds computation (ongoing work)
* fixed bugs in MiniZinc and Z3's rewriting, related to int vs bool

## 0.9.12
New:
* examples/ use notebooks with graphic visualisation whenever possible
* examples/ add pareto optimal enumeration
* new global: AllDifferentExcept0, with tests

Changed:
* minizinc: status time in seconds
* reify_rewrite: very special case, if (non-total) element elemnt-wise decomp returns false,
* flatten: avoid unnecessary sum decompositions
* SolverLookup: would select the last one if an invalid was given
* solveAll() return warning if an objective function is present
* comparing a boolean expression with an intvar crashed most solvers (#208)
* z3: bugfixes in translation
* globals: give xor a logic-based decomposition
* tests: multiple improvements, run with all solvers


## 0.9.11
FuzzTest bugfix release

* core: importing a file and adding constrains can turn problems wrongly unsatisfiable (#174)
* core: Added floordivision '//' on integer variables (#201)
* core: Wsum care (#182)
* core: cumulative can not assume args are numpy array (#178)
* core: add custom cpmpy exceptions
* core: #172 fix - memory crash on mod
* core: double negation bug #170
* globals: Enable deep copy of Cumulative global constraint (#169)

* examples: csplib prob001 car sequencepy sliding window view only numpy 1200+ (#186)
* tests: (re)active constraint tests on ortools as default solver (#180)
* tests: fix deprecation warnings
* tests: skipif when solvers not available
* tests: enable automatic tests on github
* docs: multiple fixes
* docs: add developer guidelines

* transf: Make line-expr more linear (#200)
* solvers: minizinc stopped with a non zero exit code but did not output an error message (#192)
* solvers: gurobi: timeout status fix

* tools: Example for maximal propagate set of constraints (#147)
* tools: prevent incrementality of solvers during tuning
* tools: mus tool crashes with only 1 soft constraint (#196)


## 0.9.10
New:
* tools: added MUS computation to tools
* tools: added hypaerparameter tuning (with Decaprio, CPAIOR22) to tools
* cumulative global constraint, with native ortools and decomposition 
* solvers: Z3 interface. Implemented both SAT and OPT subsolvers
* examples: CSPlib problems, with data, all runnable
* examples: IJCAI22 tutorial on 'CP as oracle' slides and notebooks
* variables: add v.clear() to easily clear variables their value

Changed:
* critical bugfix in double negation normalization
* bugfixes in solver translations
* bugfix in negation of globals normalization
* docs: improved docs on hyperparameter search and adding and testing solvers
* gurobi: fix timeout status
* bugfix in element decomposition

## 0.9.9
Stabilizing the core implementations and a knowledge compiler as solver

New:
* New example: ortools as propagator (only propagation)
* Copy method for model and constraints (and hence hash functions)
* Bools are now treated as numeric when needed
* Xor is now a global constraint (with a decomposition)
* the reify-rewrite transformation: most reifications automatically supported
* the only-numexpr-equality transformation, now generic for all solvers that want
* PySDD as solver: knowledge compiler for Boolean logic and solveAll model counting

Changed:
* Docs: many additions (direct solver access, solveAll, adding a solver guide) and improvements
* Transformation flatten has cleaner implementation
* Bugfixes in bound computations (modulo, weighted sum)
* MiniZinc: fix allequal translation, use integer division
* Reification of Elements now takes range of index variable into account
* Examples: small fixes and improvements
* Tests: more and better

## 0.9.8
An exciting 'technical' release that opens the door to
add many more solvers

* API change: unified interfaces of solvers/
* New: the gurobi MIP solver
* New: transformations/linearisation/
* More extensive testing
* PySAT: support time\_limit argument to solve

## 0.9.7
* New: s.solveAll(): convenient (efficient) solution enumeration
* New: added sum() to python\_builtins, behaves like np.sum
* Behind the scenes: add 'wsum' weighted sum operator
* bugfix for sum: always create new expression, do not modify inplace
* bugfix: allow model with only an objective

## 0.9.6
* Added tutorial video and used notebooks
    https://www.youtube.com/watch?v=A4mmmDAdusQ

* Added to examples/:
    - LP/CP contest 2021, first problem
    - wolf-goat-cabbage and n-puzzle rework
    - palindrome day problem
    - graph coloring australia (with actual map)

* Added to examples/advanced/:
    - CPMpy versions of visual examples from A. Schiendorfer
    - visual sudoku example with pytorch neural network classification
    - cost-optimal unsatisfiable subset search example
    - step-wise explanations of satisfiable problems
    - smart predict + optimize with integrated pytorch training
    - counterfactual explanations of optimisation problems
    - VRP by learning from historical data

* API change:
    - m.solve() now only returns True/False, also for optimisation
    - new: m.objective\_value(), to get the objective after solving
    - new: SolverLookup.get(solvername, model) for easy solver getting
        names also allow e.g. 'minizinc:chuffed' and 'pysat:glucose4'

* pysat: better checking of correct package installed
* pysat: automatic encoding of cardinality constraints like sum(x) >= v
* to\_cnf: more testing, some bugfixes
* ort: basic support for 'power' operator
* ort: added installation instructions for or-tools on Apple M1
* ort: bugfix in solution hinting, clear hints once
* ort: fix log callback duplicate printing
* mzn: generic fix for offset 1/0 errors
* model: better handle empty constraints and non-standard lists
* model: can now save\_to/load\_from pickle files
* bugfixes in bounds computations for modulo operator
* get\_vars: ensure the transformation returns unique elements
* requirements included minizinc, no longer the case (it is optional)
* many documentation updates

## 0.9.5
* fix bug in ort limitation check for 'modulo' operator
* mzn: better doc and check on single solution output
* various documentation updates

## 0.9.4
Major:
* re-enabled MiniZinc as a backend solver!
* reworked how solvers (and subsolvers) are accessed
    -> you can now do `model.solve(solver="minizinc:chuffed")` and the like
* added a SolverLookup.solvernames() to get supported names
* a debugging guide in the docs

Minor:
* various documentation and test updates
* some more explicit errors
* add vectorized operations that were missing (thanks Hakan)
* pysat: fix bug where constraints were duplicated
* ort: show validation error when model is invalid
* ort: work around 'xor' not being reifiable
* add missing negated_normal for 'xor'

## 0.9.3
* make progress logging work in jupyter/IPython (beta ortools feature)
* transf/get_variables now has print_variables that prints domains, for debugging with domains
* fix automatic bounds computation of auxiliary variables for abs,mul,div,pow (mostly due to negative numbers)

## 0.9.2
* pysat: tseitin encode all logical operators
* to_cnf tseitin encoding for logical operators, with tests
* better chaining of n-ary operators, fixes #39
* doc: beginner tutorial add optimisation
* doc: extend multiple solutions, minisearch, diverse solutions
* ort: add OrtSolutionPrinter and solve(solution_callback=...)
* example of diverse solutions
* vectorized 'abs' operator
* flatten: fix some bound computations

## 0.9.1
* easier hyperparameter search with `param_combinations()` helper function in cpmpy.solvers

## 0.9.0

First beta release!

* Reorganize cpmpy/ modules (not backward-compatible)
* Rework variables/constraint constructors (deprecation warnings for old constructors, will be removed with stable release)
* Updated all examples to follow new style
* Add PySAT backend, with incrementality/core extraction (only accepts CNF input for now)
* Add minimize()/maximize() to Model() and solver objects
* simpler `from cpmpy.solvers import CPM_ortools` for solver-specific use
* Add keyword arguments to solve() that configure solver-specific options
* Add example of hyperparameter gridsearch
* Updated API and user docs considerably


## 0.7.2

* get_core() work around bug in upstream ortools

## 0.7.1

* still learning the right release flow, this is a stable release

## 0.7.0b

* still learning the right release flow

## 0.7.0

### Major
* Reworked solver interface so that it is near-identical to model interface
* Or-tools interface allows unsat core extraction!
* Add MARCO MUS enumerate as example of unsat core extraction usage
* variables now take a name= argument for variable name (easier debugging)

### Enhancements
* Added more examples: bibd, npuzzle
* Added 'table' global constraint
* Added support for time\_limit when calling solve()
* Added more tests on the flattening
* Add solution hints to ortools interface
* Improved documentation

### Bugfixes
* multiple fixes and improvements in ortools interface
* fix module (thanks HakanK)
* various bugfixes

## 0.6.0

### Major

* Or-tools is now the default backend

### Enhancements
* A new `flat normal form` with `flatten_model` transformation 
* Generic global constraint decompositions
* Adding more examples
* Documentation improvements.
* Multiple bug fixes related to the integration of OR-Tools
