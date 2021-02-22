Documentation related to packaging and pushing new versions to PYPI (pip).

# Requirements
You can either use `PyPA build` or `setuptools`
- For PyPA build: `python3 -m pip install --upgrade build` this will relay on hot-installation of wheels.
- For setuptools: `python3 -m pip install --upgrade wheel`
- For both, install twine: `python3 -m pip install --upgrade twine`

# Building
As said, there are two ways to build, based on the existing packages at build time:
## PyPA build
---
After installing PyPa build, run this command 
- `python3 -m build` 

This command should output a lot of text and once completed should generate two files in the dist directory: 
```
dist/
  CP_Test_PY-0.0.1-py3-none-any.whl
  CP_Test_PY-0.0.1.tar.gz
```
The tar.gz file is a Source Archive whereas the .whl file is a Built Distribution. Newer pip versions preferentially install built distributions, but will fall back to source archives if needed. You should always upload a source archive and provide built archives for the platforms your project is compatible with.

---
## Setuptools
---
After installing PyPa build, run the following commands
-  To build the Source Archive `python3 setup.py sdist` 
-  To build the Built Distribution `python3 setup.py bdist_wheel`

The results of these commands is similar to PyPA with two files in the dist folders

---
# Uploading the distribution archives to PyPI (pip)
The first thing you’ll need to do is register an account on [Test-PyPI](https://test.pypi.org/account/register/) or [PyPI](https://pypi.org/account/register/). Then create a PyPI API token so you will be able to securely upload the project (this should happen only one time, the first time we upload to PyPI). Don’t close the page until you have copied and saved the token — you won’t see that token again.

Note To avoid having to copy and paste the token every time you upload, you can create a $HOME/.pypirc file:
```
[pypi]
  username = __token__
  password = <the token value, including the `pypi-` prefix>
[testpypi]
  username = __token__
  password = <the token value, including the `pypi-` prefix>
```

Then to upload the package, use the following commands:
- For Test-PyPI use `twine upload --repository testpypi dist/*`
- For normal PyPI use `twine upload dist/*`

---
# References
- Main python distribution documentation: [distributing-packages-using-setuptools](https://packaging.python.org/guides/distributing-packages-using-setuptools).
- A quick tutorial to deploy on: [Packaging Python Projects](https://packaging.python.org/tutorials/packaging-projects/)