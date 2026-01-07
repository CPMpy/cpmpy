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

with open("README.md", "r", encoding="utf8") as readme_file:
    long_description = readme_file.read()


solver_dependencies = {
    "ortools": ["ortools"],
    "z3": ["z3-solver"],
    "choco": ["pychoco>=0.2.1"],
    "exact": ["exact>=2.1.0"],
    "minizinc": ["minizinc"],
    "pysat": ["python-sat"],
    "gurobi": ["gurobipy"],
    "pysdd": ["pysdd"],
    "gcs": ["gcspy"],
    "cpo": ["docplex"],
    "pumpkin": ["pumpkin-solver>=0.2.1"],
    "pindakaas": ["pindakaas>=0.2.1"],
    "cplex": ["docplex", "cplex"],
}
solver_dependencies["all"] = list({pkg for group in solver_dependencies.values() for pkg in group}) 

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
        'ortools>=9.9',
        'numpy>=1.5',
        'setuptools',
        'packaging', # to check solver versions
    ],
    extras_require={
        # Solvers
        **solver_dependencies,
        # Tools
        "xcsp3": ["pycsp3", "requests", "tqdm", "matplotlib", "psutil", "filelock", "gnureadline; platform_system != 'Windows'", "pyreadline3; platform_system == 'Windows'"], # didn't add CLI-specific req since some are not cross-platform
        # Other
        "test": ["pytest", "pytest-timeout"],
        "docs": ["sphinx>=5.3.0", "sphinx_rtd_theme>=2.0.0", "myst_parser", "sphinx-automodapi", "readthedocs-sphinx-search>=0.3.2"],
    },
    entry_points={
        'console_scripts': [
            'cpmpy = cpmpy.cli:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8'
)
