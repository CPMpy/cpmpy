CPMpy: Constraint Programming and Modeling in Python
====================================================

CPMpy is a Constraint Programming and Modeling library in Python, based on numpy, with direct solver access.

Constraint Programming is a methodology for solving combinatorial optimisation problems like assignment problems or covering, packing and scheduling problems. Problems that require searching over discrete decision variables.

CPMpy allows to model search problems in a high-level manner, by defining decision variables and constraints and an objective over them (similar to MiniZinc and Essence'). You can freely use numpy functions and indexing while doing so. This model is then automatically translated to state-of-the-art solver like or-tools, which then compute the optimal answer.
   
Source code and bug reports at https://github.com/CPMpy/cpmpy

.. toctree::
   :maxdepth: 1
   :caption: Getting started:

   installation_instructions
   beginner_tutorial
   Quickstart sudoku notebook <https://github.com/CPMpy/cpmpy/blob/master/examples/quickstart_sudoku.ipynb>
   More examples <https://github.com/CPMpy/cpmpy/blob/master/examples/>

.. toctree::
   :maxdepth: 1
   :caption: User Documentation:

   solver_parameters
   multiple_solutions
   advanced_solver_features
   unsat_core_extraction
   behind_the_scenes


.. toctree::
   :maxdepth: 2
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
