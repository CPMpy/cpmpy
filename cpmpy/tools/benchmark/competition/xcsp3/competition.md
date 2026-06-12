# XCSP3 competition 2026

This document contains the installation and usage instructions for the CPMpy submission to the XCSP3 2026 competition.

## Submission

This submission is the basis for multiple submissions with different solver backends. CPMpy is a modelling system which can translate to many different solvers, 8 of which have been chosen for the XCSP3 competition. The data files and install instructions are shared (some solvers have additional installation steps). From the executable's point of view, the major difference between the submissions is the actual command used to run the executable, where the correct solver must be set. These commands are listed later on. Internally, different solver-tailored backends will be used, where the COP and CSP models get transformed as to satisfy the modelling capabilities of the selected solver target.

The CPMpy modelling system will compete in the following tracks, using the following solver backends:

| CPMpy_backend | CSP sequential | COP sequential (3') | COP sequential (30') | COP parallel |
| - | - | - | - | - |
| **cpmpy_ortools** | yes | yes | yes | yes |
| **cpmpy_exact** | yes | yes | yes | **NO** |
| **cpmpy_z3** | yes | yes | yes | **NO** |
| **cpmpy_gurobi** | yes | yes | yes | yes |
| **cpmpy_cpo** | yes | yes | yes | yes |
| **cpmpy_highs** | yes | yes | yes | yes |
| **cpmpy_scip** | yes | yes | yes | yes |
| **cpmpy_pindakaas_cadical** | yes | **NO** | yes | yes |


## Setup

Since the competition will be run on a cluster of Rocky Linux 9.5 servers, the installation steps have been talored to that particular OS (but for version 9.8, unsure if there will be any difference). Our submission is not inherently dependant on any particular OS, but some dependencies might be missing on a clean install (which are for example included in a standard Ubuntu install).

### Dependencies

CPMpy is a Python library, and thus python is the main dependency. But many of the libraries on which it depends for communicating with the different solvers, have their own dependencies, often requirering certain C libraries. The next steps should be done in order, since they need to be available when compiling python3.12 from source. When installing python3.12 through any other means, there is no guarantee that it has been build with these dependencies included (The anaconda builds do seem to include everything). 


1) C compiler & other dev tools

    Some python libraries have C dependencies and need to be able to compile them
    upon setup (when pip installing). The following installs various development tools (like g++) :

    ```bash
    sudo dnf group install "Development Tools"
    ```

2) libffi-devel

    To get `_ctypes` in python. Similarly as the previous dependency, it is required to
    build some of the python libraries. Must be installed before compiling python3.12 (if python3.12 is not already installed)
    ```bash
    sudo dnf install -y libffi-devel
    ```

3) openssl-devel 

    To get pip working with locations that require TLS/SSL.
    ```bash
    sudo dnf install -y openssl-devel 
    ```

4) ncurses-devel

    Needed for building the "readline" python library, otherwise linker will complain.
    ```bash
    sudo dnf install -y ncurses-devel
    ```

5) boost-devel

    Needed for the `Exact` solver.
    ```bash
    sudo dnf install boost-devel
    ```
5) cmake

    Needed for the `Exact` solver.
    ```bash
    sudo dnf install cmake
    ```

6) Python
    - version: 3.12(.11)

    These are the steps we used to install it:
    ```bash
    curl -O https://www.python.org/ftp/python/3.12.11/Python-3.12.11.tgz
    tar xzf Python-3.12.11.tgz 
    cd Python-3.12.11
    ./configure --enable-optimizations 
    make altinstall 
    ```   
    The last command might need 'sudo'.

    Python should now be available with the command `python3.12`

    > [!WARNING]
    > If the above dependencies are not installed at the time of building python, later installation steps for some of the solvers will fail.



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
    pip install .[z3,gurobi,cpo,scip,highs,pindakaas,cplex,xcsp3]
    pip install -r ./cpmpy/tools/xcsp3/requirements.txt
    ```

5) Finish CPO license setup

    ```
    docplex config --upgrade <install location of the CPLEX_Studio222>
    ```

### Solvers

1) Gurobi: licence

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

2) CP Optimizer: binary + license

    CP Optimizer is a commercial CP solver that requires a licence to run at its full potential.

    First get a license from the CP Optimizer's website (academic license is available):
    https://www.ibm.com/products/ilog-cplex-optimization-studio
    (for academic license: https://www.ibm.com/academic/)

    The license will be downloadable as a binary, e.g. "IBM_ILOG_CPLEX_OptStdv22.2_LIN.bin".

    Run the installer inside the binary (you can install it anywhere):
    ```bash
    chmod +x IBM_ILOG_CPLEX_OptStdv22.2_LIN.bin 
    ./IBM_ILOG_CPLEX_OptStdv22.2_LIN.bin
    ```

    As a last step, you'll need to edit a config file to point to your cplex studio install. A sample for this config has already been provided:
    ```bash
    cp cpmpy/tools/xcsp3/cpo_config.py.sample cpo_config.py
    ```
    Now fill in the install location in `cpo_config.py`.

3) Exact: build with SoPlex

    Installing the Exact solver with SoPlex, following the instructions from their docs (https://gitlab.com/nonfiction-software/exact):

    ```console
    git clone https://gitlab.com/nonfiction-software/exact.git

    cd exact
    git submodule init
    git submodule update

    mkdir soplex_build
    cd soplex_build
    cmake ../soplex -DBUILD_TESTING="0" -DSANITIZE_UNDEFINED="0" -DCMAKE_BUILD_TYPE="Release" -DBOOST="0" -DGMP="0" -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS="0" -DZLIB="0" -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    make -j 8

    cd ../build
    cmake .. -DCMAKE_BUILD_TYPE="Release" -Dsoplex="ON"
    make -j 8
    ```

    Then enable SoPlex in the Python bindings and install Exact into your environment (from the Exact repository root):

    ```console
    cd ..
    git apply --check <path_to_submission>/cpmpy/tools/benchmark/competition/xcsp3/exact-setup-soplex.patch
    git apply <path_to_submission>/cpmpy/tools/benchmark/competition/xcsp3/exact-setup-soplex.patch
    pip install .
    ```


Now we should be all set up!


## Usage

The interface of the executable is as follows:

```console
python run_benchmark.py BENCHNAME
    --runner xcsp3 
    [--seed RANDOMSEED] 
    [--time_limit TIMELIMIT] 
    [--mem_limit MEMLIMIT] 
    --verbose                   # To print output to stdout
    [--cores NBCORES]       
    [--intermediate]
    [--solver SOLVER]           # Name of the CPMpy solver backend
```

`run_benchmark.py` can be found at `./cpmpy/tools/benchmark/cpbenchy/runner/`.

The same executable supports multiple solver backends and is used for all of the submissions to the competition. The submitted cpmpy + backends are:
- `cpmpy_ortools`
- `cpmpy_exact`
- `cpmpy_z3`
- `cpmpy_gurobi`
- `cpmpy_cpo`
- `cpmpy_highs`
- `cpmpy_scip`
- `cpmpy_pindakaas_cadical`


The commands are as follows:

| **Submission** | **Command** |
| - | - |
| **`cpmpy_<solver name>`** | python ./cpmpy/tools/benchmark/cpbenchy/runner/run_benchmark.py BENCHNAME --runner xcsp3 --time_limit TIMELIMIT --mem_limit MEMLIMIT --verbose --cores NBCORES --seed RANDOMSEED --intermediate --solver `<solver name>` |
| **cpmpy_ortools** | python ./cpmpy/tools/benchmark/cpbenchy/runner/run_benchmark.py BENCHNAME --runner xcsp3 --time_limit TIMELIMIT --mem_limit MEMLIMIT --verbose --cores NBCORES --seed RANDOMSEED --intermediate --intermediate --solver ortools |
| **cpmpy_exact** | python ./cpmpy/tools/benchmark/cpbenchy/runner/run_benchmark.py BENCHNAME --runner xcsp3 --time_limit TIMELIMIT --mem_limit MEMLIMIT --verbose --cores NBCORES --seed RANDOMSEED --intermediate --solver exact |
| **cpmpy_z3** | python ./cpmpy/tools/benchmark/cpbenchy/runner/run_benchmark.py BENCHNAME --runner xcsp3 --time_limit TIMELIMIT --mem_limit MEMLIMIT --verbose --cores NBCORES --seed RANDOMSEED --intermediate --solver z3 |
| **cpmpy_gurobi** | python ./cpmpy/tools/benchmark/cpbenchy/runner/run_benchmark.py BENCHNAME --runner xcsp3 --time_limit TIMELIMIT --mem_limit MEMLIMIT --verbose --cores NBCORES --seed RANDOMSEED --intermediate --solver gurobi |
| **cpmpy_cpo** | python ./cpmpy/tools/benchmark/cpbenchy/runner/run_benchmark.py BENCHNAME --runner xcsp3 --time_limit TIMELIMIT --mem_limit MEMLIMIT --verbose --cores NBCORES --seed RANDOMSEED --intermediate --solver cpo |
| **cpmpy_highs** |python ./cpmpy/tools/benchmark/cpbenchy/runner/run_benchmark.py BENCHNAME --runner xcsp3 --time_limit TIMELIMIT --mem_limit MEMLIMIT --verbose --cores NBCORES --seed RANDOMSEED --intermediate --solver highs |
| **cpmpy_scip** | python ./cpmpy/tools/benchmark/cpbenchy/runner/run_benchmark.py BENCHNAME --runner xcsp3 --time_limit TIMELIMIT --mem_limit MEMLIMIT --verbose --cores NBCORES --seed RANDOMSEED --intermediate --solver scip |
| **cpmpy_pindakaas_cadical** | python ./cpmpy/tools/benchmark/cpbenchy/runner/run_benchmark.py BENCHNAME --runner xcsp3 --time_limit TIMELIMIT --mem_limit MEMLIMIT --verbose --cores NBCORES --seed RANDOMSEED --intermediate --solver pindakaas |

