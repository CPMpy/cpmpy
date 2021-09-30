from setuptools import find_packages, setup
import codecs
import os.path

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

setup(
    name='cpmpy',
    version=get_version("cpmpy/__init__.py"),
    author='Tias Guns',
    author_email="tias.guns@kuleuven.be",
    license='Apache 2.0',
    description='A numpy-based library for modeling constraint programming problems',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CPMpy/cpmpy",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'ortools>=5.0',
        'numpy>=1.5',
    ],
    #extra dependency, only needed if minizinc is to be used.
    extras_require={
        "FULL":  ["minizinc"],
    },
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)
