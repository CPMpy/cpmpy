CPMpy: Constraint Programming and Modeling in Python
====================================================

Constraint Programming is a methodology for solving combinatorial optimisation problems like assignment problems or covering, packing and scheduling problems. Problems that require searching over discrete decision variables.

CPMpy is a Constraint Programming and Modeling library in Python, based on numpy, with direct solver access. Key features are:

* Easy to integrate with machine learning and visualisation libraries, because decision variables are numpy arrays.
* Solver-independent: transparently translating to CP, MIP, SMT and SAT solvers
* Incremental solving and direct access to the underlying solvers
* and much more...

.. toctree::
   :maxdepth: 1
   :caption: Usage

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
    - "exact" the Exact integer linear programming solver

* Tier 3 solvers: they are work in progress and live in a pull request
    - "gcs" the Glasgow Constraint Solver

We hope to upgrade many of these solvers to higher tiers, as well as adding new ones. Reach out on github if you want to help out.


.. toctree::
   :maxdepth: 1
   :caption: Advanced guides:

   how_to_debug
   direct_solver_access
   multiple_solutions
   unsat_core_extraction
   developers
   adding_solver

Open Source
-----------

Source code and bug reports at https://github.com/CPMpy/cpmpy

CPMpy has the open-source `Apache 2.0 license <https://github.com/cpmpy/cpmpy/blob/master/LICENSE>`_ and is run as an open-source project. All discussions happen on Github, even between direct colleagues, and all changes are reviewed through pull requests.

**Join us!** We welcome any feedback and contributions. You are also free to reuse any parts in your own project. A good starting point to contribute is to add your models to the examples folder.


Are you a solver developer? We are keen to integrate solvers that have a python API on pip. If this is the case for you, or if you want to discuss what it best looks like, do contact us!


.. toctree::
   :maxdepth: 1
   :caption: API documentation:

   api/expressions
   api/model
   api/solvers
   api/transformations