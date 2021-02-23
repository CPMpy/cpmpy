# Documentation Creation by Sphinx (https://docs.readthedocs.io/en/stable/intro/getting-started-with-sphinx.html)

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


```sh
make html
```

It will create `index.html`. Open it on a browser to look. To add headings and subheadings modify the  `index.rst` and then make the html again.
## With markdown

To use markdown the documentation says we have to `pip install recommonmark` and then add the following extension in the ``conf.py``
```python
extensions = ['recommonmark']
```
# Building package Locally
First create a `setup.py` file
```python
from setuptools import setup, find_packages

setup(
    name='cpmpy',
    version='0.0.1',
    description='Python module for CP',
    author='Tias Guns',
    author_email='',
    license='MIT',
    # packages=find_packages(exclude=['cpmpy.test']),
    install_requires=['numpy'],
    # entry_points={
    #     'console_scripts': [
    #         'sapp=cpmpy.cpmpy:main' #for creating executable
    #     ]
    # }
)
```
Specify the requirements in the requirements.txt file
```
numpy>=1.20.0
```
Create a folder named `cpmodules` (or whatever) and write all the class and definition inside it within `model.py` (or whatever).
Now to install in test mode, first create a virtual environment. and then run after going to directory containing the `setup.py`
```
pip install -e .
```
To check whether the installation is successful, open python
```python
form cpmodules.model import *
```
This should work from anywhere, not only from that directory!

# Testing using tox https://tox.readthedocs.io/en/latest/
To test first create a folder named `tests` and write some functions to test.
Next create a `tox.ini` file just next to the `setup.py`.
```
# content of: tox.ini , put in same dir as setup.py
[tox]
envlist = py37,py36
[testenv]
platform = linux2|darwin    #to specify which platform
usedevelop = True
deps = pytest               # PYPI package providing pytest
commands = pytest {posargs} # substitute with tox' positional arguments
```
`tox` uses `pyest` for testing. We have to use  `usedevelop = True` for local testing. Then just run
```
tox
```
or
```
tox -e py36
```
to test on a specific environment.

If you receive ` congratulations :)` your installation is successful.
# Building Package for distribution
After successful testing build your package, using the command
```
python setup.py sdist
```
It will create a `dist` folder and inside it, you will find `tar.gz` file. This file is sharable with others and others can  `pip install` this file. 
# Creating PyPI package
