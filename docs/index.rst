
CPMpy: CP modeling made easy in Python
==========================================

CPMpy is a numpy-based light-weight Python library for conveniently modeling constraint problems in Python. It aims to connect to common constraint solving systems that have a Python API, such as MiniZinc (with solvers gecode, chuffed, ortools, picatsat, etc), or-tools through its Python API and more.

It is inspired by CVXpy, SciPy and Numberjack, and as most modern scientific Python tools, it uses numpy arrays as basic data structure.

A longer description of its motivation and architecture is in [this short paper](https://github.com/tias/cppy/blob/master/modref19_cppy.pdf)

The software is in ALPHA state, and more of a proof-of-concept really. Do send suggestions, additions, API changes, or even reuse some of these ideas in your own project!


Preface
-------

.. toctree::

   preface/overview
   preface/behind_the_scenes

API documentation
-----------------

.. toctree::

   api/expressions
   api/model
   api/variables
   api/globalconstraints
   api/solver_interfaces

Supplementary :mod:`.examples` package
--------------------------------------

.. toctree::
   examples/all_examples

