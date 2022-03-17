# Change log

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
