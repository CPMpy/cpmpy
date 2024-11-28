# TODO

A list of things in CPMpy's docs which need fixed / which are optional improvements.

A lot of proof-reading is still needed to catch all formatting mistakes (especially in the API section, as taken for the code's docstrings)


- [ ] Outdated copyright
    - whilst updated on the repo, ReadTheDocs still gives 2021 as copyright year

- [ ] Mention Exact
    - In many places where solvers get listed, Exact is not inluded (especially for things that make Exact stand out; e.g. unsat core extraction)

- [ ] Solver overview table
    - available solvers and their supported features
    e.g. incrementality, core exctraction, proof logging, parallelisation, ...

- [ ] Overview of available CPMpy tools
- [ ] "frietkot" link in ocus is down
- [ ] docs already refer to cse in "how to add a solver"
    - CSE has not been added to master yet

- [ ] Decision variables shape
    - decision variable docs say that all variables are arrays and thus have the shape attribute
    -> only NDVarArrays

- [ ] Sphinx's python autodoc annotations:
    - see Exact's interface
    - [The Python Domain — Sphinx documentation (sphinx-doc.org)](https://www.sphinx-doc.org/en/master/usage/domains/python.html#directive-py-method)

- [ ] Gurobi GRB_ENV subsolver
    document this

- [ ] Jupyter notebooks

- [ ] Use of "warning" boxes & others

