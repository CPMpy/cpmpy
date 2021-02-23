.. cpmpy documentation master file, created by
   sphinx-quickstart on Mon Feb 22 12:15:31 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Cpmpy: CP modeling made easy in Python
==========================================

Cpmpy is a numpy-based light-weight Python library for conveniently modeling constraint problems in Python. It aims to connect to common constraint solving systems that have a Python API, such as MiniZinc (with solvers gecode, chuffed, ortools, picatsat, etc), or-tools through its Python API and more.

It is inspired by CVXpy, SciPy and Numberjack, and as most modern scientific Python tools, it uses numpy arrays as basic data structure.

A longer description of its motivation and architecture is in [this short paper](https://github.com/tias/cppy/blob/master/modref19_cppy.pdf)

The software is in ALPHA state, and more of a proof-of-concept really. Do send suggestions, additions, API changes, or even reuse some of these ideas in your own project!


.. toctree::
   :maxdepth: 3
   :caption: Contents:

   structure