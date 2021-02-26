
CPMpy: CP modeling made easy in Python
==========================================

Welcome to CpMPy. Licensed under the MIT License.

CpMPy is a numpy-based light-weight Python library for conveniently modeling constraint problems in Python. It aims to connect to common constraint solving systems that have a Python API, such as MiniZinc (with solvers gecode, chuffed, ortools, picatsat, etc), or-tools through its Python API and more.

It is inspired by CVXpy, SciPy and Numberjack, and as most modern scientific Python tools, it uses numpy arrays as basic data structure.

A longer description of its motivation and architecture is in :download:`pdf <modref19_cppy.pdf>`.

The software is in ALPHA state, and more of a proof-of-concept really. Do send suggestions, additions, API changes, or even reuse some of these ideas in your own project!

Check the CP [tutorial](https://github.com/tias/cppy/blob/master/docs/overview.rst).

### Install the library

.. tutorial/how_to_install:

### Documentation

Get the full CpMPy [documentation](https://cpmpy.readthedocs.io/en/latest/). 

.. toctree::
   :maxdepth: 1
   :caption: Preface:

   preface/overview
   preface/behind_the_scenes

.. toctree::
   :maxdepth: 1
   :caption: API documentation:

   api/expressions
   api/model
   api/variables
   api/globalconstraints
   api/solver_interfaces

Supplementary :mod:`.examples` package
--------------------------------------

.. toctree::
   examples/all_examples
