CPMpy: CP modeling made easy in Python
==========================================

Welcome to CpMPy. Licensed under the MIT License.

CpMPy is a numpy-based light-weight Python library for conveniently modeling constraint problems in Python. It aims to connect to common constraint solving systems that have a Python API, such as MiniZinc (with solvers gecode, chuffed, ortools, picatsat, etc), or-tools through its Python API and more.

It is inspired by CVXpy, SciPy and Numberjack, and as most modern scientific Python tools, it uses numpy arrays as basic data structure.

A longer description of its motivation and architecture is in :download:`pdf <modref19_cppy.pdf>`.

The software is in ALPHA state, and more of a proof-of-concept really. Do send suggestions, additions, API changes, or even reuse some of these ideas in your own project!

Install the library
-------------------
.. toctree::
   :maxdepth: 2

   tutorial/how_to_install

.. toctree::
   :maxdepth: 1
   :caption: Preface:

   preface/overview
   preface/behind_the_scenes

.. toctree::
   :maxdepth: 1
   :caption: API documentation:

   api/variables
   api/expressions
   api/constraints
   api/model
   api/solver_interfaces

Supplementary :mod:`.examples` package
--------------------------------------

.. toctree::
   :caption: Examples:

   examples/all_examples

FAQ
---

**Problem**: I get the following error:


.. code-block:: python

   "IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices"

Solution: Indexing an array with a variable is not allowed by standard numpy arrays, but it is allowed by cpmpy-numpy arrays. First convert your numpy array to a cpmpy-numpy array with the `cparray()` wrapper:

.. code-block:: python
   :linenos:

   # x is a variable 
   X = IntVar(0, 3)

   # Transforming a given numpy-array **m** into a cparray
   m = cparray(m)
   
   # apply constraint
   m[X] == 8

License
-------

This library is delivered under the MIT License, (see [LICENSE](https://github.com/tias/cppy/blob/master/LICENSE)).
