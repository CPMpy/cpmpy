CPMpy: Constraint Programming and Modeling in Python
====================================================

CPMpy is a Constraint Programming and Modeling library in Python, based on numpy, with direct solver access.

Constraint Programming is a methodology for solving combinatorial optimisation problems like assignment problems or covering, packing and scheduling problems. Problems that require searching over discrete decision variables.

CPMpy allows to model search problems in a high-level manner, by defining decision variables and constraints and an objective over them (similar to MiniZinc and Essence'). You can freely use numpy functions and indexing while doing so. This model is then automatically translated to state-of-the-art solver like or-tools, which then compute the optimal answer.
   
Source code and bug reports at https://github.com/CPMpy/cpmpy

Getting started:

.. toctree::
   :maxdepth: 1
   :caption: Getting started:

   Youtube tutorial video <https://www.youtube.com/watch?v=A4mmmDAdusQ>
   beginner_tutorial
   installation_instructions
   Quickstart sudoku notebook <https://github.com/CPMpy/cpmpy/blob/master/examples/quickstart_sudoku.ipynb>
   More examples <https://github.com/CPMpy/cpmpy/blob/master/examples/>

.. toctree::
   :maxdepth: 1
   :caption: User Documentation:

   modeling
   solvers
   multiple_solutions
   how_to_debug
   solver_parameters
   unsat_core_extraction
   adding_solver
   developers


.. toctree::
   :maxdepth: 1
   :caption: API documentation:

   api/expressions
   api/model
   api/solvers
   api/transformations

Supported solvers
-----------------

CPMpy can translate to many different solvers, and even provides direct access to them.

To make clear how well supported and tested these solvers are, we work with a tiered classification:

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


FAQ
---

**Problem**: I get the following error:


.. code-block:: python

   "IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices"

Solution: Indexing an array with a variable is not allowed by standard numpy arrays, but it is allowed by cpmpy-numpy arrays. First convert your numpy array to a cpmpy-numpy array with the `cpm_array()` wrapper:

.. code-block:: python
   :linenos:

   # x is a variable 
   X = intvar(0,3)

   # Transforming a given numpy-array **m** into a CPMpy array
   m = cpm_array(m)
   
   # apply constraint
   m[X] == 8

License
-------

This library is delivered under the MIT License, (see [LICENSE](https://github.com/tias/cppy/blob/master/LICENSE)).
