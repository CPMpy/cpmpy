<div align="center">

![Github Version](https://img.shields.io/github/v/release/CPMpy/cpmpy?label=Github%20Release&logo=github)
![PyPI version](https://img.shields.io/pypi/v/cpmpy?color=blue&label=Pypi%20version&logo=pypi&logoColor=white)
![PyPI downloads](https://img.shields.io/pypi/dm/cpmpy?label=Pypi%20Downloads&logo=pypi&logoColor=white)
![Tests](https://github.com/CPMpy/cpmpy/actions/workflows/python-test.yml/badge.svg)
![Licence](https://img.shields.io/github/license/CPMpy/cpmpy?label=Licence)
</div>

---

<p align="center">
    <b>CPMpy</b>: a <b>C</b>onstraint <b>P</b>rogramming and <b>M</b>odeling library in <b>Py</b>thon, based on numpy, with direct solver access.
</p>


**Documentation: [https://cpmpy.readthedocs.io/](https://cpmpy.readthedocs.io/)**

---

### Constraint solving at your finger tips

For combinatorial problems with Boolean and integer variables. With many high-level constraints that are automatically decomposed when not natively supported by the solver.

Lightweight, [well-documented](https://cpmpy.readthedocs.io/), used in research and industry. 

Install simply with `pip install cpmpy`

### 🔑 Key Features

* **Solver-agnostic**: use and compare CP, ILP, SMT, PB and SAT solvers
* **ML-friendly**: decision variables are numpy arrays, with vectorized operations and constraints
* **Incremental solving**: assumption variables, adding constraints and updating objectives
* **Extensively tested**: large test-suite and [actively fuzz-tested](https://github.com/CPMpy/fuzz-test)
* **Tools**: for parameter-tuning, debugging, explanation generation and XCSP3 benchmarking
* **Flexible**: easy to add constraints or solvers, also direct solver access

### 🔩 Solvers

CPMpy can translate to a wide variety of constraint solving paradigms, including both commercial and open-source solvers.

* **CP Solvers**: OR-Tools (default), IBM CP Optimizer (license required), Choco, Glasgow GCS, Pumpkin, MiniZinc+solvers
* **ILP Solvers**: SCIP, Gurobi (license required), CPLEX (license required)
* **GO Solvers**: Hexaly (license required)
* **SMT Solvers**: Z3
* **PB Solvers**: Exact
* **SAT Encoders and Solvers**: PySAT+solvers, Pindakaas
* **Decision Diagrams**: PySDD

### <span style="font-family: monospace; font-size: 1.2em;">&lt;/&gt;</span> Example: flexible jobshop scheduling

An [example](https://github.com/CPMpy/cpmpy/blob/master/examples/flexible_jobshop.py) that also demonstrates CPMpy's seamless integration into the scientific Python ecosystem:

```python
# Parallel machine scheduling: a set of jobs must be scheduled, each can be run on compatible machines,
# with different duration and energy consumption. Minimize makespan and total energy consumption
import cpmpy as cp
import pandas as pd
import random; random.seed(1)

# --- Data definition ---
num_jobs = 15
num_machines = 3
# Generate some data: [job_id, machine_id, duration, energy]
data = [[jobid, machid, random.randint(2, 8), random.randint(5, 15)]
        for jobid in range(num_jobs) for machid in range(num_machines)]
df_data = pd.DataFrame(data, columns=['job_id', 'machine_id', 'duration', 'energy'])
num_compatible = len(df_data)

# Compute maximal horizon (sum of longest duration per job)
horizon = df_data.groupby("job_id")["duration"].max().sum()

# Decision `start[j]`: integer start time for each job `j`
start = cp.intvar(0, horizon, shape=num_jobs, name="start")
# Decision `end[j]`: integer end time for each job `j`
end = cp.intvar(0, horizon, shape=num_jobs, name="end")
# Decision `active[(j,m)]`: Per compatible combination, Boolean indicating if it is used
active = cp.boolvar(shape=num_compatible, name="active")

model = cp.Model()

# Mandatory jobs: each job must be assigned to exactly one compatible machine
for _, job_rows in df_data.groupby("job_id"):
    model += cp.sum(active[job_rows.index]) == 1

# Machine capacity: each machine can only process one job at a time
# also enforces Matching duration: the end time of a job is the start time + the duration on its assigned machine
for _, mach_rows in df_data.groupby("machine_id", sort=True):
    model += cp.NoOverlapOptional(
        start = start[mach_rows["job_id"]],
        end = end[mach_rows["job_id"]],
        duration = mach_rows["duration"].values,
        is_present = active[mach_rows.index],
    )

# Metric Makespan: end time of the latest job
makespan = cp.max(end)

# Metric Total energy: total energy consumed by the machines
total_energy = cp.sum(active * df_data["energy"])

# Objective: minimize makespan, and to a lesser extend also total energy consumption
model.minimize(100 * makespan + total_energy)


# --- solving and graphical visualisation ---
if model.solve():
    print(model.status())
    print("Total makespan:", makespan.value(), "energy:", total_energy.value())

    # Visualize with Plotly's excellent Gantt chart support
    import plotly.express as px
    df_solution = df_data[active.value() == True].copy()  # Select rows where active is True
    df_solution["start"] = pd.to_datetime(start[df_solution.index].value(), unit="m")
    df_solution["end"] = pd.to_datetime(end[df_solution.index].value(), unit="m")
    import plotly.io as pio; pio.renderers.default = "browser"
    px.timeline(df_solution, x_start="start", x_end="end", y="machine_id", color="job_id", text="energy").show()
else:
    print("No solution found.")
```

You can then compare the runtime of all installed solvers, or [much more](https://cpmpy.readthedocs.io/)...
```python
for solvername in cp.SolverLookup.solvernames():
    try:
        model.solve(solver=solvername, time_limit=10)  # max 10 seconds
        print(f"{solvername}: {model.status()}")
    except Exception as e:
        print(f"{solvername}: Not run -- {str(e)}")
```

### 🌳 Ecosystem

CPMpy is part of the scientific Python ecosystem, making it easy to use in Jupyter notebooks, to add visualisations, or to use it in machine learning pipelines.

Other projects that build on CPMpy:
* [XCP-explain](https://github.com/CPMpy/XCP-explain): a library for explainable constraint programming
* [PyConA](https://github.com/CPMpy/pyconA): a library for constraint acquisition
* [Fuzz-Test](https://github.com/CPMpy/fuzz-test): fuzz testing of constraint solvers
* [Sudoku Assistant](https://sudoku-assistant.cs.kuleuven.be): an Android app for sudoku scanning, solving and intelligent hints
* [CHAT-Opt demonstrator](https://chatopt.cs.kuleuven.be): translates natural language problem descriptions into CPMpy models

Also, CPMpy participated in both the [2024 and 2025 XCSP3 competition](https://www.xcsp.org/competitions/), twice making its solvers win 3 gold and 1 silver medal.

## 🔧 Library development

CPMpy has the open-source [Apache 2.0 license]( https://github.com/cpmpy/cpmpy/blob/master/LICENSE) and is run as an open-source project. All discussions happen on Github, even between direct colleagues, and all changes are reviewed through pull requests. 

Join us! We welcome any feedback and contributions. You are also free to reuse any parts in your own project. A good starting point to contribute is to add your models to the `examples/` folder.

Are you a **solver developer**? We are keen to integrate solvers that have a Python API and are on pip. Check out our [adding solvers](https://cpmpy.readthedocs.io/en/latest/adding_solver.html) documentation and contact us!


## 🙏 Acknowledgments

Part of the development received funding through Prof. Tias Guns' European Research Council (ERC) Consolidator grant, under the European Union’s Horizon 2020 research and innovation programme (grant agreement No 101002802, [CHAT-Opt](https://people.cs.kuleuven.be/~tias.guns/chat-opt.html)).

You can cite CPMpy as follows: "Guns, T. (2019). Increasing modeling language convenience with a universal n-dimensional array, CPpy as python-embedded example. The 18th workshop on Constraint Modelling and Reformulation at CP (ModRef 2019).

```
@inproceedings{guns2019increasing,
    title={Increasing modeling language convenience with a universal n-dimensional array, CPpy as python-embedded example},
    author={Guns, Tias},
    booktitle={Proceedings of the 18th workshop on Constraint Modelling and Reformulation at CP (Modref 2019)},
    volume={19},
    year={2019}
}
```

If you work in academia, please cite us. If you work in industry, we'd love to hear how you are using it. The lab of Prof. Guns is open to [collaborations and contract research](https://cpmpy.cs.kuleuven.be).
