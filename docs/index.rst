CPMpy: Constraint Programming and Modeling in Python
====================================================
CPMpy is a Constraint Programming and Modeling library in Python, based on numpy, with direct solver access.

Constraint Programming is a methodology for solving combinatorial optimisation problems like assignment problems or covering, packing and scheduling problems. Problems that require searching over discrete decision variables.

Some of CPMpy's key features are:

* Easy to integrate with machine learning and visualisation libraries, because decision variables are numpy arrays.
* Solver-independent: transparently translating to CP, MIP, SMT, SAT
* Incremental solving and direct access to the underlying solvers
* and much more...

.. toctree::
   :maxdepth: 1
   :caption: Basic usage

   modeling

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


.. toctree::
   :maxdepth: 1
   :caption: Advanced guides:

   how_to_debug
   direct_solver_access
   unsat_core_extraction
   developers
   adding_solver

Source code and bug reports at https://github.com/CPMpy/cpmpy

.. toctree::
   :maxdepth: 1
   :caption: API documentation:

   api/expressions
   api/model
   api/solvers
   api/transformations



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
