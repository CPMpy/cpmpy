# Competition

These are the installation and usage instructions for the CPMpy submission to the XCSP3 2024 competition.

## Submission

This submission is the basis for multiple solver submissions. CPMpy can translate to many different solvers, six of which have been chosen for the XCSP3 competition. The data files and install instructions are shared (some solvers have additional steps). The biggest difference is the actual command for the executable, where the correct solver must be set.

The following solvers will compete in the following tracks:

| Solver | CSP sequential | COP sequential | COP parallel | mini CSP | mini COP |
| - | - | - | - | - | - |
| OR-Tools | yes | yes | yes | yes | yes |
| Exact | yes | yes | no | yes | yes |
| Z3 | yes | yes | no | yes | yes |
| Gurobi | yes | yes | no | yes | yes |
| Minizinc : GeCode | yes | yes | no | yes | yes |
| Minizinc : Chuffed | yes | yes | no | yes | yes |


## Setup

Since the competition will be run on a cluster of CentOS 8.3 servers, the installation steps have been talored to that particular OS. Our submission is not inherently dependant on any particular OS, but some dependencies might be missing on a clean install (which are for example included in a standard Ubuntu install).

### Dependencies

CPMpy is a python library, and thus python is the main dependency. But many of the libraries on which it depends for communicating with the different solvers, have their own dependencies, often requirering certain C libraries. The next steps should be done in order, since they need to be available when compiling python3.11 from source. When installing python3.11 through any other means, there is no guarantee that it has been build with these dependencies included (The anaconda builds do seem to include everything). 


1) C compiler & other dev tools

    Some python libraries have C dependencies and need to be able to compile them
    upon setup (when pip installing). The following installs various development tools (like g++) :

    ```bash
    yum group install "Development Tools"
    ```

2) libffi-devel

    To get _ctypes in python. Similarly as the previous dependency, is required to
    build some of the python libraries. Must be installed before compiling python3.11 (if python3.11 is not already installed)
    ```bash
    yum install -y libffi-devel
    ```

3) ncurses-devel

    Needed for building the "readline" python library, otherwise linker will complain.
    ```bash
    yum install -y ncurses-devel
    ```

4) boost-devel

    Needed for the `Exact` solver.
    ```bash
    yum install boost-devel
    ```

5) libGL divers

    For some reason, the `GeCode` solver requires graphics drivers to be able to run (probably uses them for vector operations?). Without them, GeCode will complain (when running, not when in stalling) about a missing shared object file: `libEGL.so.1`. This might not be present on a headless install.

    ```bash
    yum install mesa-libGL mesa-dri-drivers libselinux libXdamage libXxf86vm libXext
    dnf install mesa-libEGL
    ```

6) Python
    - version: 3.11(.7)

    These are the steps we used to install it:
    ```bash
    wget https://www.python.org/ftp/python/3.11.7/Python-3.11.7.tgz
    tar xzf Python-3.11.7.tgz 
    cd Python-3.11.7
    ./configure --enable-optimizations 
    make altinstall 
    ```   
    Python should now be available with the command `python3.11`

    > [!WARNING]
    > If the above dependencies are not installed at the time of building python, later installation steps for some of the solvers will fail.

### Solvers

1) Minizinc


    To download Minizinc, run the following:
    ```bash
    wget https://github.com/MiniZinc/MiniZincIDE/releases/download/2.8.5/MiniZincIDE-2.8.5-bundle-linux-x86_64.tgz
    tar zxvf MiniZincIDE-2.8.5-bundle-linux-x86_64.tgz 
    ```
    Now add the `/bin` directory inside the extracted directory to `PATH`.

    E.g.:
    ```bash
    export PATH="$HOME/MiniZincIDE-2.8.5-bundle-linux-x86_64/bin/:$PATH"
    ```

2) Gurobi licence

    It might be that you already have a licence or that, depending on how the licence was acquired, the installation of the license differs. The following steps are for installing a single-person single-machine academic license, aquired through the Gurobi User Portal.

    First, get a licence from Gurobi's site. It should give you a command looking like: `grbgetkey <your licence key>`

    Next, get the license installer:

    ```bash
    wget https://packages.gurobi.com/lictools/licensetools11.0.2_linux64.tar.gz
    tar zxvf licensetools11.0.2_linux64.tar.gz 
    ```

    Now install the license:
    ```bash
    ./grbgetkey <your licence key>
    ```
    It will ask where you would like to install the license. As long as Gurobi can find the license again,
    the exact location does not matter for CPMpy.


### Installation

These are the final steps to install everything from CPMpy's side. We will create a python virtual environment and install all libraries inside it.

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
    Install Poetry (a python dependency manager)
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
python executable/main.py <BENCHNAME> 
    [-s/--seed <RANDOMSEED>] 
    [-l/--time-limit=<TIMELIMIT>] 
    [-m/--mem-limit <MEMLIMIT>] 
    [-c/--cores <NBCORES>]              
    [--solver <SOLVER>]             # Name of solver
    [--subsolver <SUBSOLVER>]       # Name of subsolver (if applicable)
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
| OR-Tools | / | python executable/main.py BENCHNAME --intermediate --cores=NBCORES --solver=ortools --mem-limit=MEMLIMIT --time-limit=TIMELIMIT --seed=RANDOMSEED | 
| Exact | / | python executable/main.py <BENCHNAME> --intermediate --cores=NBCORES --solver=exact --mem-limit=MEMLIMIT --time-limit=TIMELIMIT --seed=RANDOMSEED | 
| Z3 | / | python executable/main.py <BENCHNAME> --intermediate --cores=NBCORES --solver=z3 --mem-limit=MEMLIMIT --time-limit=TIMELIMIT --seed=RANDOMSEED | 
| Gurobi | / | python executable/main.py <BENCHNAME> --intermediate --cores=NBCORES --solver=gurobi --mem-limit=MEMLIMIT --time-limit=TIMELIMIT --seed=RANDOMSEED | 
| Minizinc | Chuffed | python executable/main.py <BENCHNAME> --intermediate --cores=NBCORES --solver=minizinc --subsolver=chuffed --mem-limit=MEMLIMIT --time-limit=TIMELIMIT --seed=RANDOMSEED | 
| Minizinc | GeCode | python executable/main.py <BENCHNAME> --intermediate --cores=NBCORES --solver=minizinc --subsolver=gecode --mem-limit=MEMLIMIT --time-limit=TIMELIMIT --seed=RANDOMSEED | 







