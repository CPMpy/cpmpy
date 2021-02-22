# Documentaion Creation by Sphinx (https://docs.readthedocs.io/en/stable/intro/getting-started-with-sphinx.html)

## Installation
Install sphinx using pip
```sh
pip install sphinx
```
## Usage
Go to the docs folder and run sphinx

```sh
sphinx-quickstart
```

```
```sh
make html
```

It will create `index.html`. Open it on a browser to look. To add headings and subheadings modify the  `index.rst` and then make the html again.
## With markdown

To use markdown the documentation says we have to `pip install recommonmark` and then add the following extension in the ``conf.py``
```python
extensions = ['recommonmark']
```
# Testing using tox https://tox.readthedocs.io/en/latest/

```
# content of: tox.ini , put in same dir as setup.py
[tox]
envlist = py27,py36

[testenv]
# install pytest in the virtualenv where commands will be executed
deps = pytest
commands =
    # NOTE: you can run any command line tool here - not just tests
    pytest

```
