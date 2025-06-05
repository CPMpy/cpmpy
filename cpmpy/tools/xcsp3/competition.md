# XCSP3 competition 2025

This document contains the installation and usage instructions for the CPMpy submission to the XCSP3 2025 competition.

## Submission

This submission is the basis for multiple submissions with different solver backends. CPMpy is a modelling system which can translate to many different solvers, seven of which have been chosen for the XCSP3 competition. The data files and install instructions are shared (some solvers have additional installation steps). From the executable's point of view, the major difference between the submissions is the actual command used to run the executable, where the correct solver must be set. These commands are listed later on. Internally, different solver-tailored backends will be used, where the COP and CSP models get transformed as to satisfy the modelling capabilities of the selected solver target.

The CPMpy modelling system will compete in the following tracks, using the following solver backends:

| CPMpy_backend | CSP sequential | COP sequential (3') | COP sequential (30') | COP parallel |
| - | - | - | - | - |
| **cpmpy_ortools** | yes | yes | yes | yes |
| **cpmpy_exact** | yes | yes | yes | no |
| **cpmpy_z3** | yes | yes | yes | no |
| **cpmpy_gurobi** | yes | yes | yes | yes |
| **cpmpy_cpo** | yes | yes | yes | yes |
| **cpmpy_mnz_gecode** | yes | yes | yes | no |
| **cpmpy_mnz_chuffed** | yes | yes | yes | no |


## Setup

Since the competition will be run on a cluster of Rocky Linux 9.5 servers, the installation steps have been talored to that particular OS (but for version 9.6, unsure if there will be any difference). Our submission is not inherently dependant on any particular OS, but some dependencies might be missing on a clean install (which are for example included in a standard Ubuntu install).

### Dependencies

CPMpy is a python library, and thus python is the main dependency. But many of the libraries on which it depends for communicating with the different solvers, have their own dependencies, often requirering certain C libraries. The next steps should be done in order, since they need to be available when compiling python3.10 from source. When installing python3.10 through any other means, there is no guarantee that it has been build with these dependencies included (The anaconda builds do seem to include everything). 


1) C compiler & other dev tools

    Some python libraries have C dependencies and need to be able to compile them
    upon setup (when pip installing). The following installs various development tools (like g++) :

    ```bash
    sudo dnf group install "Development Tools"
    ```

2) libffi-devel

    To get `_ctypes` in python. Similarly as the previous dependency, it is required to
    build some of the python libraries. Must be installed before compiling python3.10 (if python3.10 is not already installed)
    ```bash
    sudo dnf install -y libffi-devel
    ```

3) openssl-devel 

    To get pip working with locations that require TLS/SSL.
    ```bash
    sudo dnf install -y openssl-devel 
    ```

5) ncurses-devel

    Needed for building the "readline" python library, otherwise linker will complain.
    ```bash
    sudo dnf install -y ncurses-devel
    ```

5) boost-devel

    Needed for the `Exact` solver.
    ```bash
    sudo dnf install boost-devel
    ```

6) libGL divers

    The `GeCode` solver requires certain graphics drivers to be installed on the system (probably uses them for vector operations?).
   Without them, GeCode will crash (when running, not when in installing) on a missing shared object file: `libEGL.so.1`.
   This might not be present on a headless install of the OS.

    ```bash
    sudo dnf install mesa-libGL mesa-dri-drivers libselinux libXdamage libXxf86vm libXext
    sudo dnf install mesa-libEGL
    ```

7) Python
    - version: 3.10(.16)

    These are the steps we used to install it:
    ```bash
    curl -O https://www.python.org/ftp/python/3.10.16/Python-3.10.16.tgz
    tar xzf Python-3.10.16.tgz 
    cd Python-3.10.16
    ./configure --enable-optimizations 
    make altinstall 
    ```   
    Python should now be available with the command `python3.10`

    > [!WARNING]
    > If the above dependencies are not installed at the time of building python, later installation steps for some of the solvers will fail.

### Solvers

1) Minizinc

    To download Minizinc, run the following:
    ```bash
    curl -LO https://github.com/MiniZinc/MiniZincIDE/releases/download/2.9.0/MiniZincIDE-2.9.0-bundle-linux-x86_64.tgz
    tar zxvf MiniZincIDE-2.9.0-bundle-linux-x86_64.tgz 
    ```
    Now add the `/bin` directory inside the extracted directory to `PATH`.

    E.g.:
    ```bash
    export PATH="$HOME/MiniZincIDE-2.9.0-bundle-linux-x86_64/bin/:$PATH"
    ```

2) Gurobi licence

    Gurobi is a commercial MIP solver that requires a licence to run at its full potential.
    Depending on the type of licence, the installation instructions differ.
    The following instructions are tailored to installing a "Academic Named-User License", which can be aquired on the [Gurobi licence portal](https://www.gurobi.com/academia/academic-program-and-licenses).

    First, get a licence from Gurobi's site. It should give you a command looking like: `grbgetkey <your licence key>`

    Next, get the license installer:

    ```bash
    curl -O https://packages.gurobi.com/lictools/licensetools11.0.2_linux64.tar.gz
    tar zxvf licensetools11.0.2_linux64.tar.gz 
    ```

    Now install the license:
    ```bash
    ./grbgetkey <your licence key>
    ```
    It will ask where you would like to install the license. As long as Gurobi can find the license again,
    the exact location does not matter for CPMpy.

3) CP Optimizer

    CP Optimizer is a commercial CP solver that requires a licence to run at its full potential.

    First get a license from the CP Optimizer's website (academic license is available):
    https://www.ibm.com/products/ilog-cplex-optimization-studio
    (for academic license: https://www.ibm.com/academic/)

    The license will be downloadable as a binary, e.g. "cplex_studio2211.linux_x86_64.bin".

    Run the installer inside the binary (you can install it anywhere):
    ```bash
    chmod +x cplex_studio2211.linux_x86_64.bin 
    ./cplex_studio2211.linux_x86_64.bin
    ```

    Make sure all dependencies are installed in your python environment:
    ```bash
    python <path to cplex install>/python/setup.py install
    ```

    As a last step, you'll need to edit a config file to point to your cplex studio install. A sample for this config has already been provided:
    ```bash
    cp cpmpy/tools/xcsp3/cpo_config.py.sample cpo_config.py
    ```
    Now fill in the install location in `cpo_config.py`.



### Installation

These are the final steps to install everything from CPMpy's side. We will create a python virtual environment and install all libraries inside it.


1) Create python virtual environment
    ```bash
    python3.10 -m venv .venv
    ```

2) Activate python environment

    Everything should be installed in the python virtual environment. Either activate the environment using:
    ```bash
    source .venv/bin/activate
    ```
    or replace `python` with the path to the python executable and `pip` with `<path_to_python> -m pip` in each of the following steps.

3) Navigate to the root directory of this submission.

4) Install python libraries

    ```bash
    pip install .[exact,z3,gurobi,minizinc,cpo,xcsp3]
    pip install -r ./cpmpy/tools/xcsp3/requirements.txt
    ```
Now we should be all set up!



## Running code

This section will explain how to run the executable on problem instances. 

The interface of the executable is as follows:
```bash
python ./cpmpy/tools/xcsp3/xcsp3_cpmpy.py <BENCHNAME> 
    [-s/--seed <RANDOMSEED>] 
    [-l/--time-limit=<TIMELIMIT>] 
    [-m/--mem-limit <MEMLIMIT>] 
    [-c/--cores <NBCORES>]              
    [--solver <SOLVER>]             # Name of solver, can be solver:subsolver
    [--intermediate]                # If intermediate results should be reported (only for COP and a subset of solvers)
```

The same executable supports multiple solver backends and is used for all of the submissions to the competition. The submitted cpmpy + backends are:
- `cpmpy_ortools`
- `cpmpy_exact`
- `cpmpy_z3`
- `cpmpy_gurobi`
- `cpmpy_cpo`
- `cpmpy_mnz_chuffed`
- `cpmpy_mnz_gecode`

The commands are as follows:

| Submission | Command |
| - | - |
| **cpmpy_ortools** | python ./cpmpy/tools/xcsp3/xcsp3_cpmpy.py BENCHNAME --intermediate --cores=NBCORES --solver=ortools --mem-limit=MEMLIMIT --time-limit=TIMELIMIT --seed=RANDOMSEED | 
| **cpmpy_exact** | python ./cpmpy/tools/xcsp3/xcsp3_cpmpy.py BENCHNAME --intermediate --cores=NBCORES --solver=exact --mem-limit=MEMLIMIT --time-limit=TIMELIMIT --seed=RANDOMSEED | 
| **cpmpy_z3** | python ./cpmpy/tools/xcsp3/xcsp3_cpmpy.py BENCHNAME --intermediate --cores=NBCORES --solver=z3 --mem-limit=MEMLIMIT --time-limit=TIMELIMIT --seed=RANDOMSEED | 
| **cpmpy_gurobi** | python ./cpmpy/tools/xcsp3/xcsp3_cpmpy.py BENCHNAME --intermediate --cores=NBCORES --solver=gurobi --mem-limit=MEMLIMIT --time-limit=TIMELIMIT --seed=RANDOMSEED | 
| **cpmpy_cpo** | python ./cpmpy/tools/xcsp3/xcsp3_cpmpy.py BENCHNAME --intermediate --cores=NBCORES --solver=cpo --mem-limit=MEMLIMIT --time-limit=TIMELIMIT --seed=RANDOMSEED | 
| **cpmpy_mnz_chuffed** | python ./cpmpy/tools/xcsp3/xcsp3_cpmpy.py BENCHNAME --intermediate --cores=NBCORES --solver=minizinc:chuffed --mem-limit=MEMLIMIT --time-limit=TIMELIMIT --seed=RANDOMSEED | 
| **cpmpy_mnz_gecode** | python ./cpmpy/tools/xcsp3/xcsp3_cpmpy.py BENCHNAME --intermediate --cores=NBCORES --solver=minizinc:gecode --mem-limit=MEMLIMIT --time-limit=TIMELIMIT --seed=RANDOMSEED | 

