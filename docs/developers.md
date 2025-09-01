# Developer guide

CPMpy is an open source project with an [Apache 2.0 license](./license.md). We are happy for you to read and change the code as you see fit.

This page introduces how to get started with development on CPMpy itself, with a focus on sharing these changes back with us for inclusion.


## Setting up your development environment

The easiest is to use the pip to do an 'editable install' of your local CPMpy folder. 

```python
pip install --editable .
```

With that, any change you do there (including checking out different branches) is automatically used wherever you use CPMpy on your system.

CPMpy, by default, only comes bundeled with the OR-Tools solver. If you want to use any other solvers, you can install them separately or through one of the CPMpy optional dependencies.

E.g. to install the Exact solver, you can run:

```bash
pip install --editable .[exact]
```

All solvers can easily be installed at once:
```bash
pip install --editable .[all]
```

For development purposes, CPMpy provides other usefull optional dependencies.

For running the tests:
```bash
pip install --editable .[test]
```

For building the documentation:
```bash
pip install --editable .[docs]
```


## Running the test suite

We only accept pull requests that pass all the tests. In general, you want to catch situations where your changes unknowingly screwed up another part. From your local CPMpy folder, execute:

```python
python -m pytest tests/
```

This will run all tests in our `tests/` folder.

You can also run an individual test, as such (e.g. when wanting to test a new solver):

```python
python -m pytest tests/test_solvers.py
```

Due to the scale of the test suite, we recommend making use of the [pytest-xdist](https://pypi.org/project/pytest-xdist/) plugin to run the tests in parallel.

```bash
pip install pytest-xdist
python -m pytest -n auto tests/
```

When adding new functionality to CPMpy, we kindly ask you to also add a few tests cases for it. Especially when submitting a bugfix, add a test to ensure that the bug doesn't appear again in the future.



## Writing documentation

Similar to only accepting contributions that pass the test suite, we also ask you to write documentation for your changes. As for the documentation style, we follow a subset of the [Google style guide](https://google.github.io/styleguide/pyguide.html). For a reference, see the source code of [model.py](../cpmpy/model.py).

## Examples

We are always happy to include more examples of models made with CPMpy (or other usecases of the library). Simply contribute them to the [examples/](https://github.com/CPMpy/cpmpy/tree/master/examples) folder.


## Code structure

  * `tests/` contains the tests
  * `docs/` contains the documentation. Any change there is updated, with some delay, on [https://cpmpy.readthedocs.io/](https://cpmpy.readthedocs.io/)
  * `examples/` our examples, we are always happy to include more
  * `cpmpy/` the python module that you install by running `pip install cpmpy` / `pip install --editable .`	

The module is structured as such:

  * `model.py` contains the omnipresent `Model()` container
  * `exceptions.py` contains a collection of CPMpy specific exceptions
  * `expressions/` contains classes and functions that represent and create expressions (constraints and objectives)
  * `solvers/` contains CPMpy interfaces to (the Python API interface of) solvers
  * `transformations/` contains methods to transform CPMpy expressions into other CPMpy expressions
  * `tools/` contains a set of independent tools that users might appreciate.

The typical flow in which these submodules are used when programming with CPMpy is: the user creates _expressions_ which they put into a _model_ object. This is then given to a _solver_ object to solve, which will first _transform_ the original expressions into expressions that it supports, which it then posts to the Python API interface of that particular solver.

Tools are not part of the core of CPMpy. They are additional tools that _use_ CPMpy, e.g. for debugging, parameter tuning, loading formats such as XCSP3, etc.


## GitHub practices

When filing a bug, please add a small case that allows us to reproduce it. If the testcase is confidential, mail [Tias](mailto:tias.guns@kuleuven.be) directly.

Only documentation changes can be directly applied on the master branch. All other changes should be submitted as a pull request.

When submitting a pull request, make sure it passes all tests and is sufficiently documented.

When fixing a bug, you should also add a test that checks we don't break it again in the future (typically, the case from the bugreport).

We are happy to do code reviews and discuss good ways to fix something or add a new feature. So do not hesitate to create a pull request for work-in-progress code. In fact, almost all pull requests go through at least 1 revision iteration.

