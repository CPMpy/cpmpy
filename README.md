![Github Version](https://img.shields.io/github/v/release/CPMpy/cpmpy?label=Github%20Release&logo=github)
![PyPI version](https://img.shields.io/pypi/v/cpmpy?color=blue&label=Pypi%20version&logo=pypi&logoColor=white)
![PyPI downloads](https://img.shields.io/pypi/dm/cpmpy?label=Pypi%20Downloads&logo=pypi&logoColor=white)
![Tests](https://github.com/CPMpy/cpmpy/actions/workflows/python-test.yml/badge.svg)
![Licence](https://img.shields.io/github/license/CPMpy/cpmpy?label=Licence)

CPMpy is a Constraint Programming and Modeling library in Python, based on numpy, with direct solver access.

* Easy to integrate with machine learning and visualisation libraries, because decision variables are numpy arrays.
* Solver-independent: transparently translating to CP, MIP, SMT, SAT
* Incremental solving and direct access to the underlying solvers
* and much more...

**Documentation: [https://cpmpy.readthedocs.io/](https://cpmpy.readthedocs.io/)**

CPMpy is still in Beta stage, and bugs can occur. If so, please report the issue on Github!

## Open Source

CPMpy has the open-source [Apache 2.0 license]( https://github.com/cpmpy/cpmpy/blob/master/LICENSE) and is run as an open-source project. All discussions happen on Github, even between direct colleagues, and all changes are reviewed through pull requests. 

Join us! We welcome any feedback and contributions. You are also free to reuse any parts in your own project. A good starting point to contribute is to add your models to the examples folder.


Are you a solver developer? We are keen to integrate solvers that have a python API on pip. If this is the case for you, or if you want to discuss what it best looks like, do contact us!

## Teaching with CPMpy

CPMpy can be a good library for courses and projects on **modeling constrained optimisation problems**, because its usage is similar to that of other data science libraries, and because it translates to the fundamental languages of SAT, SMT, MIP, and CP transparently.

Contact Prof. Tias Guns if you are interested in, or are going to develop, teaching material using CPMpy. For example we have CPMpy snippets of part of Pierre Flener's excellent ["Modelling for Combinatorial Optimisation [M4CO]"](https://user.it.uu.se/~pierref/courses/COCP/slides/).

## Acknowledgments

Part of the development received funding through Prof. Tias Guns his European Research Council (ERC) Consolidator grant, under the European Union’s Horizon 2020 research and innovation programme (grant agreement No 101002802, [CHAT-Opt](https://people.cs.kuleuven.be/~tias.guns/chat-opt.html)).

You can cite CPMpy as follows: "Guns, T. (2019). Increasing modeling language convenience with a universal n-dimensional array, CPpy as python-embedded example. The 18th workshop on Constraint Modelling and Reformulation at CP (ModRef 2019).

```
@inproceedings{guns2019increasing,
    title={Increasing modeling language convenience with a universal n-dimensional array, CPpy as python-embedded example},
    author={Guns, Tias},
    booktitle={Proceedings of the 18th workshop on Constraint Modelling and Reformulation at CP (Modref 2019)},
    volume={19},
    year={2019}
}
```

