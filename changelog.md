# Change log

## 0.9.25

### Added
* **New solver**: IBM CP Optimizer [#576](https://github.com/CPMpy/cpmpy/pull/576)
* **New global**: `ShortTable` [#469](https://github.com/CPMpy/cpmpy/pull/469)
* `model.add()` as new default to add constraints to model (same behaviour as `+=`) [#640](https://github.com/CPMpy/cpmpy/pull/640)
* Easy install of all solvers through the `"all"` optional dependency, i.e. `pip install cpmpy[all]` [#665](https://github.com/CPMpy/cpmpy/pull/665)
* More complex variants of the Sudoku puzzle (in examples) [#577](https://github.com/CPMpy/cpmpy/pull/577)
* API summery sheet [#629](https://github.com/CPMpy/cpmpy/pull/629)

* Linearisation for `div`, `abs` and `mod` [#516](https://github.com/CPMpy/cpmpy/pull/516)
* Linearisation for subtraction [#639](https://github.com/CPMpy/cpmpy/pull/639)
* `div` linearisation for Gurobi [#593](https://github.com/CPMpy/cpmpy/pull/593)
* Native cardinality constraints for PySAT [#588](https://github.com/CPMpy/cpmpy/pull/588)
* `wsum` support for PySAT

* Support for constants in `Element` global function [#630](https://github.com/CPMpy/cpmpy/pull/630)

* `SolverLookup` now has `.supported()` and `.print_status()` to get information on the available solvers on the current system [#641](https://github.com/CPMpy/cpmpy/pull/641)
* `solveAll()` now accepts solver-specific kwargs, just like `solve()` [#582](https://github.com/CPMpy/cpmpy/pull/582)

* Optional dependencies for solvers and tools (setup.py) [#599](https://github.com/CPMpy/cpmpy/pull/599)
* Documentation for all CPMpy exceptions [#622](https://github.com/CPMpy/cpmpy/pull/622)



### Changed
* Bumped minimal Python version from 3.7 to 3.8 [#575](https://github.com/CPMpy/cpmpy/pull/575)
* `mod` and `div` now default to integer division
* Improved reading of DIMACS formatted files [#587](https://github.com/CPMpy/cpmpy/pull/587)
* Avoid max-based decomposition for `abs` if possible [#595](https://github.com/CPMpy/cpmpy/pull/595)
* Cleanup semantics of Boolean Python builtins [#602](https://github.com/CPMpy/cpmpy/pull/602)
* Performance optimisation of `simplify_bool` [#592](https://github.com/CPMpy/cpmpy/pull/592)
* Ensure `AllDifferent` decomposition is linear [#614](https://github.com/CPMpy/cpmpy/pull/614)
* Much improved README 
* Improved docs formatting, especially for the Python API docstrings [#603](https://github.com/CPMpy/cpmpy/pull/603)
* Improved API documentation of explanation tools [#512](https://github.com/CPMpy/cpmpy/pull/512)
* Better exceptions for explanation tools [#512](https://github.com/CPMpy/cpmpy/pull/512)
* Improve documentation of non-standard solver args passing for Exact [#616](https://github.com/CPMpy/cpmpy/pull/616)
* General documentation improvements [#619](https://github.com/CPMpy/cpmpy/pull/619), [#633](https://github.com/CPMpy/cpmpy/pull/633), [#634](https://github.com/CPMpy/cpmpy/pull/634)
* Skip subset of `PySAT` tests when optional dependency `pblib` is not available [#668](https://github.com/CPMpy/cpmpy/pull/668)

### Fixed
* Linearisation with boolean constants [#581](https://github.com/CPMpy/cpmpy/pull/581)
* Linearisation of `AllDifferent` with integer constants [#547](https://github.com/CPMpy/cpmpy/pull/547)
* Side conditions for `Precedence` global [#589](https://github.com/CPMpy/cpmpy/pull/589)
* `simplify_bool` on non-CPMpy expressions [#626](https://github.com/CPMpy/cpmpy/pull/626)
* Handling of negative variables in objective during linearisation [#495](https://github.com/CPMpy/cpmpy/pull/495)
* Integers in `GCC` global for Choco-solver [#646](https://github.com/CPMpy/cpmpy/pull/646)
* `NValueExceptN` for single value range [#645](https://github.com/CPMpy/cpmpy/pull/645)
* Handling of empty clauses in GCS [#662](https://github.com/CPMpy/cpmpy/pull/662)
* Missing user vars when calling `solveAll()`, resulting in incorrect number of solutions [#609](https://github.com/CPMpy/cpmpy/pull/609)
* Consistent handling of non-positive `time_limit` [#642](https://github.com/CPMpy/cpmpy/pull/642)
* Added `setuptools` to required dependencies (can be missing on some installs) [#664](https://github.com/CPMpy/cpmpy/pull/664)
* DIMACS tempfiles on Windows [#601](https://github.com/CPMpy/cpmpy/pull/601)

### Removed

* Removed support for Python version 3.7 [#575](https://github.com/CPMpy/cpmpy/pull/575)

**Full Changelog**: https://github.com/CPMpy/cpmpy/compare/v0.9.24...v0.9.25

<!-- ---------------------------------- - ---------------------------------- -->

## 0.9.24

### Release Notes

#### Enhancements and Features
- **Safening of partial functions**: New transformation to safen partial function. [#515](https://github.com/CPMpy/cpmpy/pull/515)
- **Update Choco interface with native reification**: Improved the Choco solver interface with native reification support. [#526](https://github.com/CPMpy/cpmpy/pull/526)  
- **Reals in objective function**: Added support for real numbers in the objective function. [#529](https://github.com/CPMpy/cpmpy/pull/529)  
- **Better naive grow**: Enhanced the "naive grow" strategy for better performance. [#528](https://github.com/CPMpy/cpmpy/pull/528)  
- **Blocks world**: Introduced a "blocks world" example for demonstration purposes. [#533](https://github.com/CPMpy/cpmpy/pull/533)  
- **Examples Colab links**: Added direct Colab links to examples for easier experimentation. [#553](https://github.com/CPMpy/cpmpy/pull/553)  
- **Circuit decomposition for all ranges**: Extended circuit decomposition to handle all ranges. [#424](https://github.com/CPMpy/cpmpy/pull/424)  
- **Global function in Z3 objective**: Introduced global functions in Z3 solver objectives. [#560](https://github.com/CPMpy/cpmpy/pull/560)  
- **Z3 auto subsolver**: Implemented automatic subsolver selection for Z3 when solving optimization problems. [#567](https://github.com/CPMpy/cpmpy/pull/567)  
- **Parametrize solver tests**: Streamlined solver test cases using parameterization. [#572](https://github.com/CPMpy/cpmpy/pull/572)  
- **Linearize power**: Added linearization for power operations. [#538](https://github.com/CPMpy/cpmpy/pull/538)  
- **Improve Boolean normalization for PySAT**: Enhanced normalization of Boolean terms for PySAT. [#569](https://github.com/CPMpy/cpmpy/pull/569)  

#### Performance Improvements
- **Has subexpr optimization**: Skip transformations of leaf expressions for improved efficiency. [#532](https://github.com/CPMpy/cpmpy/pull/532)  
- **Only implied speedup**: Optimized "only implied" handling for significant speedups. [#541](https://github.com/CPMpy/cpmpy/pull/541)  
- **Distribute tests over CPUs**: Distributed tests over 4 CPUs, reducing runtime from `21m30s` to `8m45s`. [#571](https://github.com/CPMpy/cpmpy/pull/571)  
- **ndvar_getitem optimization**: Improved `ndvar_getitem` by moving fetch casts to initialization. [#550](https://github.com/CPMpy/cpmpy/pull/550) 
- **Remove inline imports**: Instead use `import as x` at top of file. [#542](https://github.com/CPMpy/cpmpy/pull/542)  

#### Bug Fixes
- **Remove broadcast in min/max**: Fixed issues when forwarding to built-in min/max functions. [#536](https://github.com/CPMpy/cpmpy/pull/536)  
- **Convert numpy array in Table constraint**: Ensured proper conversion of NumPy arrays to lists in Table constraints. [#540](https://github.com/CPMpy/cpmpy/pull/540)  
- **Clear values on UNSAT**: Added functionality to clear variable values when UNSAT is detected. [#523](https://github.com/CPMpy/cpmpy/pull/523)  
- **Fix cpm_array with order='F'**: Resolved issues with `cpm_array()` when using column-major order. [#555](https://github.com/CPMpy/cpmpy/pull/555)  
- **Car sequencing index fix**: Corrected indexing issues in the car sequencing problem. [#565](https://github.com/CPMpy/cpmpy/pull/565)  

#### Code Quality and Maintenance
- **Improve exception messages**: Enhanced clarity of exception messages and removed unused imports. [#539](https://github.com/CPMpy/cpmpy/pull/539)  
- **Edits to the docs**: Updated documentation for clarity and completeness. [#530](https://github.com/CPMpy/cpmpy/pull/530)    
- **Gurobi license check**: Separated Gurobi license checks into a distinct process. [#566](https://github.com/CPMpy/cpmpy/pull/566)  
- **Standardize solver version checks**: Unified approach to checking solver version compatibility. [#568](https://github.com/CPMpy/cpmpy/pull/568)  
- **Update requirements**: Upped our minimal python requirement from 3.6 to 3.7. [#573](https://github.com/CPMpy/cpmpy/pull/573)

**Full Changelog**: https://github.com/CPMpy/cpmpy/compare/v0.9.23...v0.9.24

<!-- ---------------------------------- - ---------------------------------- -->

## 0.9.23

Quick release, because we want the updated tools to be available.

### What's Changed
* Extension to tools: MARCO and SMUS 
* Added tests for incremental solving and fixed incemental solving with objective in Exact
* Cumulative decomposition fix when capacity was numpy integer.

**Full Changelog**: https://github.com/CPMpy/cpmpy/compare/v0.9.22...v0.9.23

<!-- ---------------------------------- - ---------------------------------- -->

## 0.9.22

### What's New
* New solver: GlasgowConstraintSolver (GCS)
* Upgraded to Exact 2
* Minizinc print: easily extract MiniZinc and FlatZinc text from CPMpy model.
* Update TEMPLATE.py to make it clearer how to add new solvers.
* SolverLookup gives clear error message in stead of returning None
* allow kwargs in Model.solve()
* call python builtins for sum, abs, min and max without expressions in the arguments.
* All solvers now have a native_model() function, to allow native solver access.
* It's now possible to name multiple new variables at once, by providing the names in a list.
* Linearize transformation can now rewrite modulo (if multiplication is supported)
* Fix behaviour of "all", "any", "max", "min", "sum", "prod" on higher dimensional NDVarArrays (maintain dimensionality)
* Value function of expressions now always returns a python integer, where it could sometimes be a numpy integer.
* Fixed performance issue where all solver vars where seen as user vars when solving with MiniZinc

### Documentation
* Overall improvement of documentation
* update documentation of 'comparison' transformation

### New Contributors
Thanks to 2 new contributors!
* [@ThomSerg](https://github.com/ThomSerg) and [@sin3000x](https://github.com/sin3000x)

**Full Changelog**: https://github.com/CPMpy/cpmpy/compare/v0.9.21...v0.9.22

<!-- ---------------------------------- - ---------------------------------- -->

## 0.9.21

### New Global constraints:
* Increasing, Decreasing, IncreasingStrict, DecreasingStrict
* Lexicographical ordering constraints: LexLess, LexLessEq, LexChainLess, LexChainLessEq
* Scheduling constraints Precedence and NoOverlap
* Closed version of GCC
* AllDiffExceptN, AllEqualExceptN 
* Nvalues except n
* Negative table
* Among

### Bug Fixes:
* count nested in a wsum crashed MiniZinc [#461](https://github.com/CPMpy/cpmpy/issues/461)
* AllDifferentExcept0 now correctly works with MiniZinc
* User variables that get simplified away during transformation steps are now saved.
* Add missing case in simplify bool transformation.

### Quality of life
* Removed type restriction for InDomain
* Extending automatic testsuite
* Check if minizinc executable is installed

**Full Changelog**: https://github.com/CPMpy/cpmpy/compare/v0.9.20...v0.9.21

<!-- ---------------------------------- - ---------------------------------- -->

## 0.9.20

### What's Changed
* Choco is now a tier 2 solver, available to use after installing the pychoco package! 
* new DIMACS parser and writer
* SolverLookup is now a classmethod for easier use of custom solvers.
* Fixed a bug where expression bounds didn't get rounded to integers when creating an intvar
* Added a warning when expressions have non-integer bounds, as these will be rounded
* Fixed a bug in our helper function is_bool, now also recognises our BoolVal objects
* Updated our ortools and minizinc version requirements.

**Full Changelog**: https://github.com/CPMpy/cpmpy/compare/v0.9.19..v0.9.20

<!-- ---------------------------------- - ---------------------------------- -->

## 0.9.19

### What's Changed
* Update on tools/subsets, add mcs and mss tools with grow-variants.
* Full propagation with exact
* Adding NValue global constraint
* Minizinc result now saved in solver object, this allows to access the solver statistics.

### Documentation
* Update docs for tools.
* Docs on solver statistics
* Format solver api, and add missing links in docs
* Update version and copyright date
* Remove bug where python comment (#) got interpreted as a header.

### Bug Fixes
* Properly handle reified global constraints for Minizinc
* Correctly handle global constraints with list-arguments of length 1
* Add missing edge case in flatten
* Type check table constraint first argument, cannot be a constant.

**Full Changelog**: https://github.com/CPMpy/cpmpy/compare/v0.9.18...v0.9.19

<!-- ---------------------------------- - ---------------------------------- -->

## 0.9.18
Minor release with some bugfixes, we are trying to do monthly releases now, so they will be more concise.

### What's new?
* get_bounds() helper function now works on arrays and lists, returning arrays (lists) of bounds
* Pysdd added to our automated GitHub tests
* Pysdd does not support Xor anymore.

### Bugfixes
* Fixed Cumulative bug when capacity was a numpy int.
* Cumulative now works in Minizinc with only one task.
* Docs look good again
* Corrected default parameter value in docs
* Fixed visualisations in Nonogram and Room assignment notebook examples
* Adopted the new ORtools 9.8 names of tuneable parameters.

## New Contributors
* [@KennyVDV](https://github.com/KennyVDV) made their first contribution by adding a new example: chess_position

<!-- ---------------------------------- - ---------------------------------- -->

## 0.9.17
Some new solvers supported at tier 3, update to our transformations and bugfixes.

### Solver tier updates:
* Choco and SCIP our now available as tier 3 solvers in separate pull requests.

### New transformations:
* Canonical comparison, puts comparisons in a canonical normal form.
* only_bv_reifies: the bool var is sent to the left part and the boolexpr to the right part in reifications. (split out of only_bv_implies)
* Only_implies: renamed what's left from only_bv_implies, that being removing all '==' double reifications.

### What else is new?
* Allow description for a constraint
* Updated linearize transformation
* small improvements to our docs
* solution hints in Exact
* New, more efficient Xor decomposition
* added QuickXplain to our tools.
* Did some performance tests and optimized our flatten_constraint transformation by not applying distributivity of disjunctions over conjunctions.

### Bugfixes:
* mark Inverse as supported in OR-tools
* removed unnecessary use of simplify bool in minizinc and z3
* fix possible overflow in bound computations
* Solution hinting fix, when giving N-D variables and values
* Solution hinting remove python 3.9 exclusive code
* Bugfixes in MUS-tool
* 
## 0.9.16
One of our most substantial releases so far, with special focus on extending and improving the
transformations for all allowed input and all solvers.

### Solver tier updates:
* MiniZinc and Z3 are now Tier1 solvers (passes all internal tests, passes our bigtest suit, will be fuzztested in the near future)
* new solver: Exact in Tier2 (passes all internal tests, might fail on edge cases in bigtest)
  Currently Exact is not supported on Windows

### Revamped documentation
* simpler readme on github, with badges
* simpler index file on readthedocs
* consolidated scattered documentation into one prominent 'modeling.html' file on readthedocs

### Global constraints
* Global functions (Minimum, Maximum, Element, Count, Abs) now separated from Global constraints in the class diagram
* New global constraint: InDomain, to specify non-interval domains.
* Abs is now a global
* Multi-dimensional element constraint supported

### New transformations:
* decompose_in_tree. Unsupported globals now get decomposed at tree-level.
* simplify_boolean, that simplifies boolean (and equivalent-to-Boolean) expressions.
* push_down_negation. Pushes down the not-operator so that it only occurs at the leaves of the expression tree

### What else is new?
* Warning messages for common beginner mistakes.
* type-checks in operator constructor and in global constraints/functions.
* our NDVarArray now support NumPy-like method 'prod', and supports the axis argument for min, max, sum, prod, any and all
* Updated tuner tool: added a (random) grid search tuner
* pysat: accept reified cardinality constraints
* z3: handle mixed integer/boolean expressions
* Add explicit requirements for jupyter notebooks examples
* Add n-queens 1000 example.
* solver interfaces: make `__add__()` directly call transform, simpler and more explicit (#312)

### Bugfixes:
* Allow numpy booleans in boolvals
* negating boolval fix
* check for bv == iv in normalized boolexpr and only_bv_implies
* Inverse decomposition with non vararrays
* Minizinc integer bounds fix
* Minizinc False literal now gets correctly translated.


## 0.9.15
Re-release of 0.9.14 due to github action pip-release screw-up

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
