<p align="center">
    <b>CPMpy</b>: a <b>C</b>onstraint <b>P</b>rogramming and <b>M</b>odeling library in <b>Py</b>thon, based on numpy, with direct solver access.
</p>

<div align="center">

![Github Version](https://img.shields.io/github/v/release/CPMpy/cpmpy?label=Github%20Release&logo=github)
![PyPI version](https://img.shields.io/pypi/v/cpmpy?color=blue&label=Pypi%20version&logo=pypi&logoColor=white)
![PyPI downloads](https://img.shields.io/pypi/dm/cpmpy?label=Pypi%20Downloads&logo=pypi&logoColor=white)
![Tests](https://github.com/CPMpy/cpmpy/actions/workflows/python-test.yml/badge.svg)
![Licence](https://img.shields.io/github/license/CPMpy/cpmpy?label=Licence)
</div>


**Documentation: [https://cpmpy.readthedocs.io/](https://cpmpy.readthedocs.io/)**

---

### Constraint solving at your finger tips

For combinatorial optimisation problems with Boolean and integer variables. With many high-level constraints that are automatically decomposed as needed for the solver.

Lightweight, [well-documented](https://cpmpy.readthedocs.io/), used in research and industry. 

### 🔑 Key Features

* **Solver-agnostic**: use and compare CP, MIP, SMT, PB and SAT solvers
* **ML-friendly**: decision variables are numpy arrays, vectorized operators and constraints
* **Incremental solving**: assumption variables, adding constraints and updating objectives
* **Extensively tested**: large test-suite and [actively fuzz-tested](https://github.com/CPMpy/fuzz-test)
* **Tools**: for parameter-tuning, debugging and explanation generation
* **Flexible**: easy to add constraints or solvers, also direct solver access
* **Open Source**: [Apache 2.0 license](https://github.com/cpmpy/cpmpy/blob/master/LICENSE)

### 🔩 Solvers

CPMpy can translate to a wide variaty of constraint solving paradigms, including both commercial and open-source solvers.

* **CP Solvers**: OR-Tools (default), IBM CP Optimizer (license required), Choco, Glasgow GCS, MiniZinc+solvers
* **MIP Solvers**: Gurobi (license required), IBM CPLEX (license required)
* **SMT Solvers**: Z3
* **PB Solvers**: Exact
* **SAT Solvers**: PySAT+solvers, PySDD

### 🌳 Ecosystem

CPMpy is part of the scientific Python ecosystem, making it easy to use it in Jupyter notebooks, to add visualisations with matplotlib, or to use it in a machine learning pipeline.

Other projects that build on CPMpy:
* [PyConA](https://github.com/CPMpy/pyconA): a cpmpy-based library for constraint acquisition
* [XCP-explain](https://github.com/CPMpy/XCP-explain): a library for explainable constraint programming
* [Fuzz-Test](https://github.com/CPMpy/fuzz-test): fuzz testing of constraint solvers
* [Sudoku Assistant](https://sudoku-assistant.cs.kuleuven.be): an Android app for sudoku scanning, solving and intelligent hints
* [CHAT-Opt demonstrator](https://chatopt.cs.kuleuven.be): translates natural language problem descriptions into CPMpy models

Also, CPMpy participated in the [2024 XCSP3 competition](https://www.xcsp.org/competitions/), making its solvers win 3 gold and 1 silver medal.

## 🔧 Library development

CPMpy has the open-source [Apache 2.0 license]( https://github.com/cpmpy/cpmpy/blob/master/LICENSE) and is run as an open-source project. All discussions happen on Github, even between direct colleagues, and all changes are reviewed through pull requests. 

Join us! We welcome any feedback and contributions. You are also free to reuse any parts in your own project. A good starting point to contribute is to add your models to the `examples/` folder.

Are you a **solver developer**? We are keen to integrate solvers that have a Python API on pip. Check out our [adding solvers](https://cpmpy.readthedocs.io/en/latest/adding_solver.html) documentation and contact us!

## 🙏 Acknowledgments

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

If you work in academia, please cite us. If you work in industry, we'd love to hear how you are using it. The lab of Prof. Guns is open to collaborations and contract research.
