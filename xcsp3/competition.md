# Competition

These are the instructions for the XCSP3 2024 competition.

## Setup

Since the competition will be run on a cluster of CentOS 8.3 servers, the installation steps have been talored to that particular OS. Our submission is not inherently dependant on any particular OS, but some dependencies might be missing on a clean install (which are for example included in a standard Ubuntu install).

### Dependencies

1) C compiler & other dev tools
    ```bash
    yum group install "Development Tools"
    ```

2) libffi-devel

    To get _ctypes in python (must be installed when compiling python3.11, if python3.11 is not already installed)
    ```bash
    yum install -y libffi-devel
    ```

3) ncurses-devel

    Needed for building the "readline" python library, otherwise linker will complain.
    ```bash
    yum install -y ncurses-devel
    ```

4) Python
    - version: 3.11(.7)

    These are the steps we used to install it:
    ```bash
    wget https://www.python.org/ftp/python/3.11.7/Python-3.11.7.tgz
    tar xzf Python-3.11.7.tgz 
    cd Python-3.11.7
    ./configure --enable-optimizations 
    make altinstall 
    ```   

### Installation

1) Enter competition directory
    ```bash
    # cpmpy/xcsp3
    cd xcsp3
    ```

2) Create python virtual environment
    ```bash
    python3.11 -m venv .venv
    ```

3) Activate python environment

    Everything should be installed in the python virtual environment. Either activate the environment using:
    ```bash
    source .venv/bin/activate
    ```
    or replace `python` with the path to the python executable and `pip` with its respective path to the executable inside the venv in each of the following steps.

4) Install python libraries

    Install GitPython (as to automatically clone the pycsp3 repo in the correct location)
    ```bash
    pip install GitPython
    ```
    Install Poetry
    ```bash
    pip install poetry
    ```
    Run the installer script
    ```bash
    python installer.py
    ```
    Install all remaining python libraries
    ```bash
    poetry lock --no-update
    poetry install
    ```
Now we should be all set up!

## Running code

This section will explain how to run the executable on problem instances. 

The interface of the executable is as follows:
```bash
python executable/main.py <benchname> 
    [-s/--seed <RANDOMSEED>] 
    [-l/--time-limit=<TIMELIMIT>] 
    [-m/--mem-limit <MEMLIMIT>] 
    [-c/--cores <NBCORES>]              
    [--solver <SOLVER>]             # Name of solver
    [--subsolver <SUBSOLVER>]       # Name of subsolver
    [--intermediate]                # If intermediate results should be reported (only for COP and a subset of solvers)
```

The executable supports multiple solvers and is used for multiple submissions to the competition. The submitted solvers are:
- ortools
- exact
- z3
- gurobi
- minizinc:chuffed
- minizinc:gecode

The commands are as follows:

| Solver | Subsolver | Command |
| - | - | - |
| OR-Tools | / | python executable/main.py BENCHNAME --intermediate --cores=\NBCORES --profiler --solver=ortools --mem-limit=MEMLIMIT --time-limit==TIMELIMIT | 
| Exact | / | python executable/main.py <BENCHNAME> --intermediate --cores=NBCORES --profiler --solver=exact --mem-limit=MEMLIMIT --time-limit=TIMELIMIT | 
| Z3 | / | python executable/main.py <BENCHNAME> --intermediate --cores=NBCORES --profiler --solver=z3 --mem-limit=MEMLIMIT --time-limit=TIMELIMIT | 
| Gurobi | / | python executable/main.py <BENCHNAME> --intermediate --cores=NBCORES --profiler --solver=gurobi --mem-limit=MEMLIMIT --time-limit=TIMELIMIT | 
| Minizinc | Chuffed | python executable/main.py <BENCHNAME> --intermediate --cores=NBCORES --profiler --solver=minizinc --subsolver=chuffed --mem-limit=MEMLIMIT --time-limit=TIMELIMIT | 
| Minizinc | GeCode | python executable/main.py <BENCHNAME> --intermediate --cores=NBCORES --profiler --solver=minizinc --subsolver=gecode --mem-limit=MEMLIMIT --time-limit=TIMELIMIT | 







