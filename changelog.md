# Change log

## 1.0.0

### Added

* **IO** module with file format readers and writers [#842](https://github.com/CPMpy/cpmpy/pull/842)
* **Datasets**: PyTorch-compatible dataset class providing single-line access to datastes from the CO community [#900](https://github.com/CPMpy/cpmpy/pull/900) [#1037](https://github.com/CPMpy/cpmpy/pull/900):
    * XCSP3
    * JSPLib
    * PSPLib rcpsp
    * MIPLib
    * MaxSAT Eval
    * OPB
    * SAT
    * Nurse rostering
* **New Solvers** 
    * **HiGHS** open source ILP solver [#868](https://github.com/CPMpy/cpmpy/pull/868)
    * **SCIP** open source Mixed Integer Programming (MIP) and Mixed Integer Nonlinear Programming (MINLP) solver [#412](https://github.com/CPMpy/cpmpy/pull/412) [#955](https://github.com/CPMpy/cpmpy/pull/955) [#982](https://github.com/CPMpy/cpmpy/pull/982)
* **New transformations** 
    * Positive decomposition [#980](https://github.com/CPMpy/cpmpy/pull/980) [#1006](https://github.com/CPMpy/cpmpy/pull/1006)
    * Decompose linear [#836](https://github.com/CPMpy/cpmpy/pull/836) [#1059](https://github.com/CPMpy/cpmpy/pull/1059)
* **New Globals** 
    * Multi-dimensional element `NDElement` [#926](https://github.com/CPMpy/cpmpy/pull/926) [#1018](https://github.com/CPMpy/cpmpy/pull/1018)
    * FloatSum to still enable floats in objective [#957](https://github.com/CPMpy/cpmpy/pull/957)
    * Markov Decision Diagram (MDD) [#924](https://github.com/CPMpy/cpmpy/pull/924)
    * Converted multiplication to a global function [#850](https://github.com/CPMpy/cpmpy/pull/850)
    * `CumulativeOptional`, `NoOverlapOptional` [#676](https://github.com/CPMpy/cpmpy/pull/676)
* MyPy static type checking [#864](https://github.com/CPMpy/cpmpy/pull/864) [#867](https://github.com/CPMpy/cpmpy/pull/867)
    * Global constraint typing [#871](https://github.com/CPMpy/cpmpy/pull/871) [#873](https://github.com/CPMpy/cpmpy/pull/873) [#874](https://github.com/CPMpy/cpmpy/pull/874)
    * Global functions and variables [#877](https://github.com/CPMpy/cpmpy/pull/877)
    * `BoolVal`/`Comparison`/`Operator` in core [#878](https://github.com/CPMpy/cpmpy/pull/878)
    * `NDVarArray` [#886](https://github.com/CPMpy/cpmpy/pull/886)
    * `Expression` init [#894](https://github.com/CPMpy/cpmpy/pull/894) 
    * `BoolExprLike` type [#941](https://github.com/CPMpy/cpmpy/pull/941)
    * `Min`/`Max` [#965](https://github.com/CPMpy/cpmpy/pull/965)
    * Nested constraints and `cpm_array` input [#1023](https://github.com/CPMpy/cpmpy/pull/1023)
    * Other [#901](https://github.com/CPMpy/cpmpy/pull/901) [#918](https://github.com/CPMpy/cpmpy/pull/918) [#907](https://github.com/CPMpy/cpmpy/pull/907) [#934](https://github.com/CPMpy/cpmpy/pull/934) [#958](https://github.com/CPMpy/cpmpy/pull/958)
* Support `Regular` global natively for MiniZinc [#952](https://github.com/CPMpy/cpmpy/pull/952)
* Linearize reified variables [#855](https://github.com/CPMpy/cpmpy/pull/855)[ #860](https://github.com/CPMpy/cpmpy/pull/860)
* Custom and typed CSEMap object for Common Subexpression Elimination [#917](https://github.com/CPMpy/cpmpy/pull/917)
* Add encoding variables as expressions to CSE [#781](https://github.com/CPMpy/cpmpy/pull/781)
* IIS-based MUS algorithm [#880](https://github.com/CPMpy/cpmpy/pull/880) [#971](https://github.com/CPMpy/cpmpy/pull/971)
* Native MUS computation for Exact [#909](https://github.com/CPMpy/cpmpy/pull/909)
* Multi-instance tuner [#757](https://github.com/CPMpy/cpmpy/pull/757)
* Add verbosity parameter to tuners [#771](https://github.com/CPMpy/cpmpy/pull/771)
* 'subsolvers' filter argument for `SolverLookup.supported()` [#1030](https://github.com/CPMpy/cpmpy/pull/1030) 
* Decorator for non-strict variable name checks [#839](https://github.com/CPMpy/cpmpy/pull/839)
* `Description` class for human-readable metadata in `Expression` [#903](https://github.com/CPMpy/cpmpy/pull/903)
* Aspirational unit tests [#883](https://github.com/CPMpy/cpmpy/pull/883)
* Instructions on python-cov [#937](https://github.com/CPMpy/cpmpy/pull/937)
* Sudoku variant examples [#777](https://github.com/CPMpy/cpmpy/pull/777)
* Template docstrings [#724](https://github.com/CPMpy/cpmpy/pull/724)
* Dev script to time transformations [#853](https://github.com/CPMpy/cpmpy/pull/853) [848febd](https://github.com/CPMpy/cpmpy/commit/848febd9f8831cbf6796ae0175d6547ba679b1ab)
* Dev script to extract release notes [#671](https://github.com/CPMpy/cpmpy/pull/671)

### Breaking changes

With this v1.0.0 release, a lot has changed to CPMpy's internal workings. Many transformations have been completely rewritten, internal API signatures have changed and datastructures have been replaced by new ones. We split the breaking changes into two groups: changes that affect what and how you can model (with instructions on how to update your code), and changes to CPMpy's internals, which you will only notice if your code uses those internals directly or if you saved models to `.pickle` files.

#### Changes to the modeling API — how to update your code

* **Float objectives are now expressed with the new `FloatSum` global function** [#957](https://github.com/CPMpy/cpmpy/pull/957). Float coefficients are no longer allowed inside regular expressions: multiplying a decision variable with a float constant (e.g. `0.5 * x`) raises a `TypeError`, and `Model.minimize()`/`Model.maximize()` only accept integer-valued expressions.
    - *If you had a float objective*: use `FloatSum` and pass it **directly to a solver object's** `minimize()`/`maximize()` (supported by OR-Tools, Gurobi, CPLEX, SCIP, HiGHS, Z3, MiniZinc and Hexaly):
      ```python
      obj = cp.FloatSum([0.5, 1.5], [x, y])        # replaces 0.5*x + 1.5*y
      s = cp.SolverLookup.get("ortools", model)    # model contains only the constraints
      s.minimize(obj)
      s.solve()
      print(obj.value())  # read the objective from the FloatSum;
                          # s.objective_value() stays None when the optimum is not integral
      ```
      Note that `FloatSum` is not an `Expression`: it cannot be nested inside constraints or other expressions.
    - *If you had float coefficients in constraints*: rescale them to integers (e.g. replace `0.5*x + 1.5*y <= 2` by `x + 3*y <= 4`).
* **The deprecated lowercase constraint aliases `alldifferent()`, `allequal()` and `circuit()` are removed** [#905](https://github.com/CPMpy/cpmpy/pull/905) [#1050](https://github.com/CPMpy/cpmpy/pull/1050) (deprecated since 0.9.0). They are no longer exported from the `cpmpy` namespace and the functions themselves have been removed from `cpmpy.expressions.globalconstraints`; use the global constraint classes `AllDifferent`, `AllEqual` and `Circuit` instead.
* **Other functions deprecated since the 0.9.x series are removed** [#1050](https://github.com/CPMpy/cpmpy/pull/1050):
    - `BoolVar()`, `IntVar()` and `cparray()`: use `boolvar()`, `intvar()` and `cpm_array()` instead.
    - `Model.deepcopy()` and `Expression.deepcopy()`: use `copy.deepcopy()` instead.
    - `cpmpy.transformations.negation.negated_normal()`: use `recurse_negation()` (or `push_down_negation()` on the full expression tree) instead.
    - `cpmpy.transformations.get_variables.vars_expr()`: use `get_variables()` instead.
    - `cpmpy.solvers.utils.get_supported_solvers()` and the `builtin_solvers` list: use `SolverLookup.supported()` and `SolverLookup.get(name)` instead.
* **Multiplication is now a global function instead of an operator** [#850](https://github.com/CPMpy/cpmpy/pull/850). `x * y` creates a `Multiplication(x, y)` global function (still named `"mul"`) and `Operator("mul", ...)` can no longer be constructed. Modeling with `*` is unaffected, but code that inspects expressions with `isinstance(expr, Operator)` no longer matches multiplications; use `isinstance(expr, Multiplication)` instead.
* **`Element` only accepts 1-dimensional arrays** [#926](https://github.com/CPMpy/cpmpy/pull/926). Indexing a multi-dimensional array now creates the new `NDElement` global function. Keep indexing with comma-separated indices (`Arr[i,j]`) or construct `NDElement(Arr, [i,j])` directly; code that constructed `Element` with a flattened index, or that checks `isinstance(expr, Element)`, must be updated accordingly.
* **`NDVarArray` is no longer an `Expression` subclass** [#886](https://github.com/CPMpy/cpmpy/pull/886). Variable arrays (as returned by `intvar(..., shape=...)`, `boolvar(shape=...)` and `cpm_array()`) are now plain `numpy.ndarray` subclasses. All modeling functionality is unaffected (vectorized operators, `.sum()`, `.min()`/`.max()`, `.any()`/`.all()`, `.value()`, `.implies()`, indexing), but code that treats an array as an expression must be updated: `isinstance(arr, Expression)` now returns `False`, and `arr.args`, `arr.name` and `arr.is_bool()` no longer exist.
* **Expression arguments are read-only tuples** [#894](https://github.com/CPMpy/cpmpy/pull/894). `expr.args` now returns a `tuple` instead of a `list`, so it can no longer be modified in place (e.g. `expr.args.append(...)` or `expr.args[0] = ...`). Construct a new expression instead, or use `expr.update_args(new_args)` for an explicit in-place update.
* **`Table`, `ShortTable` and `NegativeTable` require a rectangular table of integers and a flat array argument** [#895](https://github.com/CPMpy/cpmpy/pull/895). The table is converted to, and stored as, a two-dimensional integer `numpy` array; ragged tables and non-integer entries now raise an error. The `array` argument is no longer flattened: a nested Python list of variables now raises an error (a multi-dimensional `NDVarArray` is still accepted and reshaped internally); flatten nested lists yourself, e.g. with `cpmpy.expressions.utils.flatlist`.
* **`Minimum` and `Maximum` no longer flatten nested lists** [#965](https://github.com/CPMpy/cpmpy/pull/965). Constructing e.g. `Minimum([[x,y],[z]])` no longer works (it creates an invalid expression that fails when solving); pass a flat sequence of expressions instead, e.g. `Minimum([x,y,z])` or `Minimum(arr.flat)`. Other globals with variable-length argument lists (`AllDifferent`, `AllEqual`, `Xor`, `Increasing`, ..., as well as `cp.min`/`cp.max`/`cp.sum` on arrays) are unaffected and still flatten their input.
* **`Cumulative` and `NoOverlap` no longer store `end` in their `.args` when it is not given** [#830](https://github.com/CPMpy/cpmpy/pull/830). Code that unpacks `.args` (e.g. `start, dur, end, demand, cap = c.args`) must first check `len(c.args)`; when no `end` was given, the end times are implicitly `start + duration`.

#### Changes to CPMpy's internals — only relevant if you use internals directly or saved models to pickle files

Due to the changes below, models saved to a `.pickle` file with CPMpy < 1.0.0 will no longer load (or will fail when used) in v1.0.0. Pickle files are tied to the CPMpy version that created them: re-run your model-building code under v1.0.0 and save again. For longer-term storage, consider writing models to a solver-independent text format with the new IO module.

* All `Expression` arguments are now stored as immutable tuples instead of lists [#894](https://github.com/CPMpy/cpmpy/pull/894), and the `has_subexpr()` cache is an eagerly-initialised attribute [#895](https://github.com/CPMpy/cpmpy/pull/895).
* Expressions created by `x * y` are `Multiplication` global functions instead of `Operator("mul", ...)` [#850](https://github.com/CPMpy/cpmpy/pull/850); multi-dimensional indexing creates `NDElement` instead of a flattened `Element` [#926](https://github.com/CPMpy/cpmpy/pull/926).
* `Table`-like constraints store their table as a 2D `numpy` array and keep `NDVarArray` arguments as-is instead of converting them to Python lists [#895](https://github.com/CPMpy/cpmpy/pull/895).
* `Cumulative` and `NoOverlap` store 4 resp. 2 arguments when no `end` is given, instead of a `None` placeholder [#830](https://github.com/CPMpy/cpmpy/pull/830).
* `set_description()` now stores a `Description` object in `expr._description`; the old `expr.desc` attribute no longer exists [#903](https://github.com/CPMpy/cpmpy/pull/903).
* The transformations have been largely rewritten in a new, typed pattern:
    - The deprecated `decompose_global()` and `do_decompose()` are removed; use `decompose_in_tree()` [#835](https://github.com/CPMpy/cpmpy/pull/835), which now also accepts custom (positive) decompositions [#929](https://github.com/CPMpy/cpmpy/pull/929) [#980](https://github.com/CPMpy/cpmpy/pull/980) [#1006](https://github.com/CPMpy/cpmpy/pull/1006).
    - The `csemap=` argument of transformations expects the new `CSEMap` object instead of a plain `dict` [#917](https://github.com/CPMpy/cpmpy/pull/917).
    - Several transformations gained parameters or dedicated objective-variants (`flatten_constraint(..., do_simplify=)`, `decompose_objective`, `push_down_negation_objective`, `safen_objective`, `decompose_linear`, `linearize_reified_variables`, ...); consult the `cpmpy.transformations` documentation when upgrading code that calls transformations directly.
* Solver interfaces: the internal `_varmap` is keyed by variable name instead of by variable object [#990](https://github.com/CPMpy/cpmpy/pull/990), and `solver_var()` of the SAT-based interfaces (PySAT, Pindakaas) consistently returns Boolean literals only [#1017](https://github.com/CPMpy/cpmpy/pull/1017).
* The deprecated `objective_value=` parameter of `SolverInterface._solve_return()` is removed, as is the internal helper `cpmpy.transformations.get_variables._uniquify()` [#1050](https://github.com/CPMpy/cpmpy/pull/1050).

#### Minor behavior changes, including bug fixes that may affect code relying on the old behavior

* **`Expression <op> ndarray` now broadcasts correctly** [#1035](https://github.com/CPMpy/cpmpy/pull/1035). Using a CPMpy expression on the left-hand side of an operator with a numpy array on the right (e.g. `x + np.array([1,2,3])`) now broadcasts element-wise and returns an array of expressions, just like the mirrored `ndarray <op> Expression` always did. Code relying on the old (faulty) single-expression result must be updated.
* **`value()` returns `None` for partially-assigned global constraints** [#872](https://github.com/CPMpy/cpmpy/pull/872). `Xor`, `Cumulative` and `Circuit` now return `None` from `.value()` when some of their arguments have no value (consistent with other expressions), instead of raising an error or computing an incorrect result.
* **Variable values are cleared after an unsatisfiable solve** [#1001](https://github.com/CPMpy/cpmpy/pull/1001). With Exact and Pindakaas, `.value()` of variables now returns `None` after an UNSAT solve call, instead of returning stale values from a previous (satisfiable) solve.
* **`cp.sum()` over a single expression returns that expression itself**, instead of wrapping it in a single-argument sum expression.
* **Expression printing has been refactored** [#893](https://github.com/CPMpy/cpmpy/pull/893). The exact output of `str(expr)` / `repr(expr)` can differ slightly from previous versions; do not rely on the textual form of expressions.

### Deprecated

Old names keep working for now (with a `DeprecationWarning` where applicable), but will be removed in a future release — switch to the new ones:

* **DIMACS tooling moved to the new IO module** [#842](https://github.com/CPMpy/cpmpy/pull/842); `cpmpy.tools.dimacs` is kept as a backward-compatible wrapper around `cpmpy.tools.io.dimacs`:
    - `read_dimacs(fname)` is deprecated; use `cpmpy.tools.io.load_dimacs(...)`, which also accepts the DIMACS content as a string or an open file object.
    - `write_dimacs(model, fname)` moved to `cpmpy.tools.io.write_dimacs(model, path)`. Note that the second parameter was renamed (`fname` → `path`) and that the `p cnf` header line is no longer written by default; pass `p_header=True` to restore it.
* **XCSP3 loading**: `cpmpy.tools.xcsp3.read_xcsp3()` is deprecated; use `cpmpy.tools.io.load_xcsp3()`, which accepts a path, string content or an open file object.
* **`XCSP3Dataset` moved to the new datasets module** [#900](https://github.com/CPMpy/cpmpy/pull/900): import it from `cpmpy.tools.datasets` (it remains re-exported from `cpmpy.tools.xcsp3` for backward compatibility).
* More generally, the new IO module provides `cpmpy.tools.io.load()` and `cpmpy.tools.io.write()` as one-stop entry points that automatically select the format (DIMACS, WCNF, OPB, XCSP3, ...) based on the file extension.
* **`SolverLookup.solvernames()`** now emits a `DeprecationWarning`; use `SolverLookup.supported()` instead [#1050](https://github.com/CPMpy/cpmpy/pull/1050). Similarly, the deprecated `_toplevel`/`nested` arguments of `decompose_in_tree()` and `_toplevel`/`_nbc` of `no_partial_functions()` now emit a `DeprecationWarning` and are ignored, instead of raising an assertion error.

### Widened and extended APIs (non-breaking)

* `solve()` now accepts a `display=...` callback that reports intermediate solutions during optimisation (supported for OR-Tools, Gurobi, GCS, CP Optimizer, HiGHS and Hexaly), like `solveAll()` already did [#561](https://github.com/CPMpy/cpmpy/pull/561).
* `solve(assumptions=...)` accepts any iterable of Boolean literals, not just a list [#712](https://github.com/CPMpy/cpmpy/pull/712).
* Global constraint and function constructors are formally typed with the new `ExprLike`/`ListLike` type aliases and accept any list-like (list, tuple, numpy array, `NDVarArray`) of expressions or constants, including numpy integers [#871](https://github.com/CPMpy/cpmpy/pull/871) [#873](https://github.com/CPMpy/cpmpy/pull/873) [#874](https://github.com/CPMpy/cpmpy/pull/874) [#877](https://github.com/CPMpy/cpmpy/pull/877).
* `SolverLookup.supported()` gained a `subsolvers=` flag to optionally exclude subsolver names from the list [#1030](https://github.com/CPMpy/cpmpy/pull/1030).
* The parameter tuners can tune over multiple problem instances at once [#757](https://github.com/CPMpy/cpmpy/pull/757) and take a `verbose=` level [#771](https://github.com/CPMpy/cpmpy/pull/771).
* Several globals gained a positive-context decomposition (`decompose_positive()`) for use in linear/SAT contexts [#980](https://github.com/CPMpy/cpmpy/pull/980) [#1006](https://github.com/CPMpy/cpmpy/pull/1006), and `AllDifferent` a linear one (`decompose_linear()`) [#836](https://github.com/CPMpy/cpmpy/pull/836).
* For advanced users: `Expression.implies(..., simplify=)`, `flatten_constraint(..., do_simplify=)` and `decompose_in_tree(..., decompose_custom=, decompose_custom_positive=)` expose new optional behavior.

### Internal improvements

* Improvements to toplevel safening [#1026](https://github.com/CPMpy/cpmpy/pull/1026)
* New pattern for `simplify_boolean` [#959](https://github.com/CPMpy/cpmpy/pull/959)
* `push_down_negation` before `decompose`[#916](https://github.com/CPMpy/cpmpy/pull/916)
* Improved MUS for Gurobi [#993](https://github.com/CPMpy/cpmpy/pull/993) [#1016](https://github.com/CPMpy/cpmpy/pull/1016)
* Improved `GCC` decomposition [#1007](https://github.com/CPMpy/cpmpy/pull/1007)
* Save `lhs` and `rhs` of reification to `CSEMap` [#1010](https://github.com/CPMpy/cpmpy/pull/1010)
* Reduction technique for MDD [#949](https://github.com/CPMpy/cpmpy/pull/949) 
* Use order encoding for inequalities [#994](https://github.com/CPMpy/cpmpy/pull/994)
* Linear decomposition for `Table` using `MDD` [#945](https://github.com/CPMpy/cpmpy/pull/945)
* Fast path solver var [#995](https://github.com/CPMpy/cpmpy/pull/995)
* Canonicalize inequalities [#996](https://github.com/CPMpy/cpmpy/pull/996)
* Refactor `decompose` in typed pattern [#929](https://github.com/CPMpy/cpmpy/pull/929)
* Simple `solver_vars` improvements [#992](https://github.com/CPMpy/cpmpy/pull/992)
* Canonicalize `neq` comparison in `CSEMap` [#969](https://github.com/CPMpy/cpmpy/pull/969)
* Avoid making auxiliary variables in `safening` [#935](https://github.com/CPMpy/cpmpy/pull/935)
* Loop optimisations [#928](https://github.com/CPMpy/cpmpy/pull/928)
* Refactor `push_down_negation` [#914](https://github.com/CPMpy/cpmpy/pull/914)
* Make `expr._has_subexpr` an attribute [#895](https://github.com/CPMpy/cpmpy/pull/895)
* Simplify `Circuit` and `Inverse` decompositions [#889](https://github.com/CPMpy/cpmpy/pull/889)
* Refactor safening transformation [#875](https://github.com/CPMpy/cpmpy/pull/875)
* PySAT pb reification optimisation; let PySAT handle conditionals [#783](https://github.com/CPMpy/cpmpy/pull/783)
* Optimize `has_subexpr` for tables [#596](https://github.com/CPMpy/cpmpy/pull/596)
* Refactor `decompose_in_tree` [#835](https://github.com/CPMpy/cpmpy/pull/835)

### Changed

* Remove very old deprecated code [#1050](https://github.com/CPMpy/cpmpy/pull/1050)
* Improve docs for testing a (new) solver [#1031](https://github.com/CPMpy/cpmpy/pull/1031)
* Standardize reporting intermediate solutions [#561](https://github.com/CPMpy/cpmpy/pull/561)
* Remove `frozendict` dependency [#1019](https://github.com/CPMpy/cpmpy/pull/1019)
* Varname in varmap instead of variable object [#990](https://github.com/CPMpy/cpmpy/pull/990)
* GCS refresh install docs, sanitise variable names with commas [#973](https://github.com/CPMpy/cpmpy/pull/973) (thanks [Ciaran](https://github.com/ciaranm))
* Update incrementality test [#976](https://github.com/CPMpy/cpmpy/pull/976)
* Convert assumptions to list [#712](https://github.com/CPMpy/cpmpy/pull/712)
* Modernize examples
    * Counterfactual [#931](https://github.com/CPMpy/cpmpy/pull/931)
    * CP explanations [#943](https://github.com/CPMpy/cpmpy/pull/943)
    * Diverse solutions [#944](https://github.com/CPMpy/cpmpy/pull/944)
    * Maximal propagate [#953](https://github.com/CPMpy/cpmpy/pull/953)
    * Stepwise explanations [#954](https://github.com/CPMpy/cpmpy/pull/954)
    * Use of `NoOverlapOptional` in flex jobshop example [#947](https://github.com/CPMpy/cpmpy/pull/947)
* Print display in solver interface superclass [#921](https://github.com/CPMpy/cpmpy/pull/921)
* Small docs/solvers update to be more complete [#920](https://github.com/CPMpy/cpmpy/pull/920)
* Modern `__init__` files [#905](https://github.com/CPMpy/cpmpy/pull/905)
* `linearize` doc update [#898](https://github.com/CPMpy/cpmpy/pull/898)
* Use is None ruff warning [#904](https://github.com/CPMpy/cpmpy/pull/904)
* Remove import \* from cpmpy/ tests/ and docs/ [#890](https://github.com/CPMpy/cpmpy/pull/890)
* Don't download examples data files during testsuite [#899](https://github.com/CPMpy/cpmpy/pull/899)
* Refactor infix printing in `Operator` for improved readability [#893](https://github.com/CPMpy/cpmpy/pull/893)
* Seperate GitHub action runners [#885](https://github.com/CPMpy/cpmpy/pull/885)
* Upgrade pindakaas to v0.5.0 & re-enable BVA for CaDiCal [#852](https://github.com/CPMpy/cpmpy/pull/852) [#869](https://github.com/CPMpy/cpmpy/pull/869)
* Upgrade Z3 lower version to v5.0.0 for [upstream bugfix](https://github.com/CPMpy/cpmpy/issues/1036) [#1052](https://github.com/CPMpy/cpmpy/pull/1052)
* Upgrade Pumpkin interface to v0.3.0 [#854](https://github.com/CPMpy/cpmpy/pull/854)
* OR-Tools and Python version bumps [#847](https://github.com/CPMpy/cpmpy/pull/847)
* Remove solver version upper limits [#848](https://github.com/CPMpy/cpmpy/pull/848)
* Convert TestCase to plain pytest classes [#818](https://github.com/CPMpy/cpmpy/pull/818)


### Fixed

* Remove 'mul' from supported for CPLEX [#1045](https://github.com/CPMpy/cpmpy/pull/1045)
* Throw `NotImplementedError` for base classes `GlobalConstraint` and `GlobalFunction` [#1047](https://github.com/CPMpy/cpmpy/pull/1047)
* XCSP3 runner 'verbose' option [#962](https://github.com/CPMpy/cpmpy/pull/962)
* Implication with single bool in lhs [#1043](https://github.com/CPMpy/cpmpy/pull/1035)
* `NDVarArray` vectorized ops to use NumPy broadcasting [#1043](https://github.com/CPMpy/cpmpy/pull/1034)
* Decomposition bugs in `Regular` and `MDD` [#1039](https://github.com/CPMpy/cpmpy/pull/1039)
* Remove trivial wrapping of bool in linearize_constraint [#1042](https://github.com/CPMpy/cpmpy/pull/1042)
* Trivial decompose crash with 0 constraints [#1041](https://github.com/CPMpy/cpmpy/pull/1041)
* Flattening of wsum remove in-place edits [#1040](https://github.com/CPMpy/cpmpy/pull/1040)
* Fix `Expr OP NDArr` by doing `NDArr ROP Expr` [#1035](https://github.com/CPMpy/cpmpy/pull/1035)
* Solver var consistent return type [#1017](https://github.com/CPMpy/cpmpy/pull/1017)
* Enforce version upper limit on HiGHS due to upstream bug [#1032](https://github.com/CPMpy/cpmpy/pull/1032)
* Enforce version upper limit on Choco due to upstream bug [#1013](https://github.com/CPMpy/cpmpy/pull/1013)
* Missing RC2 depdendencies in setup.py [#1028](https://github.com/CPMpy/cpmpy/pull/1028)
* `Table` subexpression check [#1029](https://github.com/CPMpy/cpmpy/pull/1029)
* Mark inconsistent tests as 'flaky' [#1024](https://github.com/CPMpy/cpmpy/pull/1024)
* Fix tests and update `requires_solver` docs [#1002](https://github.com/CPMpy/cpmpy/pull/1002)
* Edge case in `canonical_comparison` [#1014](https://github.com/CPMpy/cpmpy/pull/1014)
* Bounds for `pow` and `mod` [#1012](https://github.com/CPMpy/cpmpy/pull/1012)
* Fix URL PSPLib [#1004](https://github.com/CPMpy/cpmpy/pull/1004)
* Read integers with `round` instead of `int` for ILP solvers [#884](https://github.com/CPMpy/cpmpy/pull/884)
* Set stale values to None after infeasability (Exact and pindakaas) [#1001](https://github.com/CPMpy/cpmpy/pull/1001)
* Upstream criptominisat fix, remove suppression [#978](https://github.com/CPMpy/cpmpy/pull/978)
* Re-export `make_assump_model` from `tools.explain` for legacy code [#974](https://github.com/CPMpy/cpmpy/pull/974)
* Edge case in tuner where time limit can become negative [#970](https://github.com/CPMpy/cpmpy/pull/970)
* Use `get_or_make_var` for end task variables [#964](https://github.com/CPMpy/cpmpy/pull/964)
* Don't store `end` args when not given for `Cumulative` and `NopOverlap` [#830](https://github.com/CPMpy/cpmpy/pull/830) 
* MiniZinc `time_limit` deprecation warning [#927](https://github.com/CPMpy/cpmpy/pull/927) 
* Remove duplicate `Comparison` handling [#951](https://github.com/CPMpy/cpmpy/pull/951) 
* Forgotten comma in `supported_globals` OR-Tools [#942](https://github.com/CPMpy/cpmpy/pull/942)
* `Xor` simplification in decomposition [#908](https://github.com/CPMpy/cpmpy/pull/908)
* Fix `self +=` pattern [#930](https://github.com/CPMpy/cpmpy/pull/930)
* Warnings in testsuite [#919](https://github.com/CPMpy/cpmpy/pull/919)
* Re-introduce deprecated variable functions [#915](https://github.com/CPMpy/cpmpy/pull/915)
* Resolve duplicate test names [#863](https://github.com/CPMpy/cpmpy/pull/863)
* Update expression test to not randomly fail [#865](https://github.com/CPMpy/cpmpy/pull/865)
* Fix solution values in int2bool [#857](https://github.com/CPMpy/cpmpy/pull/857)
* Skip tests if Pumpkin not supported [#856](https://github.com/CPMpy/cpmpy/pull/856)
* Make tests solver-independent [#779](https://github.com/CPMpy/cpmpy/pull/779)
* `test_transf_comp` without increasing counters [#638](https://github.com/CPMpy/cpmpy/pull/638)
* Update value checks on `Xor`, `Cumulative` and `Circuit` [#872](https://github.com/CPMpy/cpmpy/pull/872)

**Full Changelog**: https://github.com/CPMpy/cpmpy/compare/v0.10.1...v1.0.0

## 0.10.1

### Fixed

This is a hotfix release due to a bug discovered in bounds computation.

* Bounds for pow and mod (cherry picked for this hotfix) [#1012](https://github.com/CPMpy/cpmpy/pull/1012)

**Full Changelog**: https://github.com/CPMpy/cpmpy/compare/v0.10.0...v0.10.1

## 0.10.0

### Added

* **New solver**: Rc2 MaxSAT solver [#729](https://github.com/CPMpy/cpmpy/pull/729)
* Expended `to_cnf` using `pumpkin` encoding backend [#782](https://github.com/CPMpy/cpmpy/pull/782)
* Linearisation of multiplication between Boolean and Integer [#769](https://github.com/CPMpy/cpmpy/pull/769)
* Solution callback for Hexaly [#787](https://github.com/CPMpy/cpmpy/pull/787), [#809](https://github.com/CPMpy/cpmpy/pull/809)
* Start of solver parametrised testsuite [#817](https://github.com/CPMpy/cpmpy/pull/817)
* Documentation overview of solver capabilities [#728](https://github.com/CPMpy/cpmpy/pull/728)
* Typehints and documentation update for global constraints [#812](https://github.com/CPMpy/cpmpy/pull/812)
* Typehints for Solve and SolveAll [#775](https://github.com/CPMpy/cpmpy/pull/775)
* OCUS in tools, allowing meta-constraints on MUSes [#698](https://github.com/CPMpy/cpmpy/pull/698)
* Fix supported solver version ranges [#816](https://github.com/CPMpy/cpmpy/pull/816)
* Transversal / hitting set example (contributed by @nandorsieben) [#790](https://github.com/CPMpy/cpmpy/pull/790)
* Nurserostering benchmark  [#789](https://github.com/CPMpy/cpmpy/pull/789)
* Prediction+optimisation example of scheduling surgeries under uncertainty [5a87c5e](https://github.com/CPMpy/cpmpy/commit/5a87c5ec464394a952b166f93076788f70f4e2ce)
* Decision-focused learning example [#621](https://github.com/CPMpy/cpmpy/pull/621)

### Internal improvements

* Globals define their own negation [#703](https://github.com/CPMpy/cpmpy/pull/703)
* Change `cp.sum(*iterable, **kwargs)` to `cp.sum(iterable, **kwargs)` [#756](https://github.com/CPMpy/cpmpy/pull/756)
* Division and Modulo as global functions [#807](https://github.com/CPMpy/cpmpy/pull/807)
* Refactor decompose for global functions [#793](https://github.com/CPMpy/cpmpy/pull/793)
* Refactor and update Cumulative and NoOverlap constraints [#694](https://github.com/CPMpy/cpmpy/pull/694)

### Changed

* Throw `ModuleNotFoundError` when module is not installed [#825](https://github.com/CPMpy/cpmpy/pull/825)

### Fixed

* Fix support for newest OR-Tools 9.15 release [#821](https://github.com/CPMpy/cpmpy/pull/821)
* Check Hexaly license before using solver [#826](https://github.com/CPMpy/cpmpy/pull/826)
* Fix `to_cnf` clause bypass issue [#824](https://github.com/CPMpy/cpmpy/pull/824)
* Fix bug handle pdk unsat with conditions [#811](https://github.com/CPMpy/cpmpy/pull/811)
* Missing packaging dependency in setup.py [#813](https://github.com/CPMpy/cpmpy/pull/813)
* Abs constraint handle None  [#794](https://github.com/CPMpy/cpmpy/pull/794)
* Timeout under assumptions for Exact [#805](https://github.com/CPMpy/cpmpy/pull/805)
* Support for expressions in start, duration and end in CPO [#802](https://github.com/CPMpy/cpmpy/pull/802)
* Consistent version checks [#792](https://github.com/CPMpy/cpmpy/pull/792)
* Missing constraint tags for Pumpkin [#799](https://github.com/CPMpy/cpmpy/pull/799)
* Z3 negate maximisation objective [#786](https://github.com/CPMpy/cpmpy/pull/786)
* Scaled Booleans in Pumpkin interface [#776](https://github.com/CPMpy/cpmpy/pull/776)

**Full Changelog**: https://github.com/CPMpy/cpmpy/compare/v0.9.29...v0.10.0

## 0.9.29

### Fixed

This is a hotfix release due to external breaking changes in the default solver backend.

* Pin solver versions (not merged, cherry picked for this hotfix) [#816](https://github.com/CPMpy/cpmpy/pull/816)

**Full Changelog**: https://github.com/CPMpy/cpmpy/compare/v0.9.28...v0.9.29

## 0.9.28

This is a very small release with a hotfix for multi-dimensional indexing where the index is a decision variable.

### Fixed

* Hotfix for single expression in multi-dimensional indexing [#772](https://github.com/CPMpy/cpmpy/pull/772)

**Full Changelog**: https://github.com/CPMpy/cpmpy/compare/v0.9.27...v0.9.28

## 0.9.27

### Added

* **New solver**: CPLEX ILP solver [#583](https://github.com/CPMpy/cpmpy/pull/583), [#762](https://github.com/CPMpy/cpmpy/pull/762), [#763](https://github.com/CPMpy/cpmpy/pull/763)
* **New solver**: Hexaly local search solver [#718](https://github.com/CPMpy/cpmpy/pull/718)
* **New tool**: CPMpy cli (get version information on all subsolvers) [#693](https://github.com/CPMpy/cpmpy/pull/693)
* Use of CSE in `flatten_objective` [#730](https://github.com/CPMpy/cpmpy/pull/730)
* Constraint tagging for `pumpkin` [#720](https://github.com/CPMpy/cpmpy/pull/720)
* Solver capability overview in docs [d8d3a2d](https://github.com/CPMpy/cpmpy/commit/d8d3a2d0f78438aa0f0da6909d74b5a505b5f8a5)
* PSP-lib dataset and loader [#701](https://github.com/CPMpy/cpmpy/pull/701)
* Safen name collision between user and aux variables [#731](https://github.com/CPMpy/cpmpy/pull/731)

### Changed

* Update `pindakaas` to version 0.2.1 [#753](https://github.com/CPMpy/cpmpy/pull/753)
* Exclude slow examples from testset [#746](https://github.com/CPMpy/cpmpy/pull/746)
* Safe variable names [#731](https://github.com/CPMpy/cpmpy/pull/731)
* SolverInterface consistency improvements and added tests [#726](https://github.com/CPMpy/cpmpy/pull/726)
* Improve testing of examples [#651](https://github.com/CPMpy/cpmpy/pull/651)
* Modernize shebangs of examples [#644](https://github.com/CPMpy/cpmpy/pull/644)
* Linearize improvements [e721b0a](https://github.com/CPMpy/cpmpy/commit/e721b0ab960be632c525bfbbf493876031ca9d98)
* Update waterfall (added automated int-to-bool) [710ec42](https://github.com/CPMpy/cpmpy/commit/710ec42fc34a014084e0f6bbe8d1e5eed72c3dcf)
* Collection of improvements to GitHub README

### Fixed

* `Precedence` global decomposition [#742](https://github.com/CPMpy/cpmpy/pull/742)
* Minor bugs in examples [#739](https://github.com/CPMpy/cpmpy/pull/739)
* `gnureadline` typo in `setup.py` [#722](https://github.com/CPMpy/cpmpy/pull/722)
* Non-contiguous array handling in `cpm_array` [#738](https://github.com/CPMpy/cpmpy/pull/738)
* Remove simplification of nullifying arguments from constructors [#725](https://github.com/CPMpy/cpmpy/pull/725)
* Skip psplib in testset due to random failure on GitHub actions [7a15d07](https://github.com/CPMpy/cpmpy/commit/7a15d07962610e3844d358a05c7c419e110f110c)
* (Pin Pumpkin solver git commit [#719](https://github.com/CPMpy/cpmpy/pull/719)) outdated by [#720](https://github.com/CPMpy/cpmpy/pull/720)

**Full Changelog**: https://github.com/CPMpy/cpmpy/compare/v0.9.26...v0.9.27

## 0.9.26

### Added

* **New solver**: Pumpkin LCG solver (by the ConSol Lab at TU Delft) [#669](https://github.com/CPMpy/cpmpy/pull/669)
* **New solver**: Pindakaas library for transforming pseudo Boolean to CNF [#600](https://github.com/CPMpy/cpmpy/pull/600)
* **New global**: `Regular` [#677](https://github.com/CPMpy/cpmpy/pull/677)
* **New tool**: Tooling for the XCSP3 format: dataset, loader, CLI, benchmarking [#597](https://github.com/CPMpy/cpmpy/pull/597)
* Common Subexpression Elimination (CSE) with normalized expressions [#679](https://github.com/CPMpy/cpmpy/pull/679)
* Integer to boolean direct encoding [#653](https://github.com/CPMpy/cpmpy/pull/653)
* Solution hinting for Gurobi [#691](https://github.com/CPMpy/cpmpy/pull/691)
* Solution callback for CP Optimizer [#682](https://github.com/CPMpy/cpmpy/pull/682)
* Acces to (installed) solver status/version: call `.version()` on a specific solver or on `SolverLookup` to get an overview. Or use `cp.SolverLookup.print_version()` for pretty printing. [#628](https://github.com/CPMpy/cpmpy/pull/628)
* Support for canonicalisation of subtraction expressions [#686](https://github.com/CPMpy/cpmpy/pull/686) 
* Allow use of boolean constants in weighted sum [#711](https://github.com/CPMpy/cpmpy/pull/711)
* Added testset for incremental assumptions
* Catch beginner mistakes linked to incorrect usage of boolean expression [#660](https://github.com/CPMpy/cpmpy/pull/660)

### Changed

* Refactoring of int2bool/pysat [#714](https://github.com/CPMpy/cpmpy/pull/714)
* Updated instructions for adding a new solver with request to add solver to overview table
* Improve efficiency `only_implies`: avoid retransform if no subexpression [#680](https://github.com/CPMpy/cpmpy/pull/680)
* Consistent state reporting of `.solve()` across all solvers [#545](https://github.com/CPMpy/cpmpy/pull/545)
* Improved docs and error messages for `canonical_comparison` [#678](https://github.com/CPMpy/cpmpy/pull/678)

### Fixed

* Support special cases for Xor global (boolean constants, single argument) [#717](https://github.com/CPMpy/cpmpy/pull/717)
* Missing `optimal` exit status for GCS [#705](https://github.com/CPMpy/cpmpy/pull/705)
* Missing solution for empty csp when enumerating [#674](https://github.com/CPMpy/cpmpy/pull/674)
* Use of constants in globals [#700](https://github.com/CPMpy/cpmpy/pull/700)
* CSE edge-case in `only_numexpr_equality` [#696](https://github.com/CPMpy/cpmpy/pull/696)
* Choco `shorttable` star symbol data type [#697](https://github.com/CPMpy/cpmpy/pull/697)
* Ensure python-native expression values to prevent unexpected behaviour with numpy constants [#695](https://github.com/CPMpy/cpmpy/pull/695)
* Correctly handle numpy integers in `eval_comparison` [#683](https://github.com/CPMpy/cpmpy/pull/683)
* CP Optimizer fix cumulative with zero-duration task [#681](https://github.com/CPMpy/cpmpy/pull/681)
* Fix incorrect decomposition of `Inverse` global constraint [#673](https://github.com/CPMpy/cpmpy/pull/673)

**Full Changelog**: https://github.com/CPMpy/cpmpy/compare/v0.9.25...v0.9.26

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
