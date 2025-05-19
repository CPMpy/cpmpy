# Flexible job-shop: a set of jobs must be run, each can be run on any of the machines,
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

# Compute maximal horizon (crude upper bound) and number of alternatives
horizon = df_data.groupby('job_id')['duration'].max().sum()
num_alternatives = len(df_data.index)
assert list(df_data.index) == list(range(num_alternatives)), "Index must be default integer (0,1,..)"


# --- Decision variables ---
start = cp.intvar(0, horizon, name="start", shape=num_alternatives)
end   = cp.intvar(0, horizon, name="end", shape=num_alternatives)
active = cp.boolvar(name="active", shape=num_alternatives)

# --- Constraints ---
model = cp.Model()

# Each job must have one active alternative
for job_id, group in df_data.groupby('job_id'):
    model += (cp.sum(active[group.index]) == 1)

# For all jobs ensure start + dur = end (also for inactives, thats OK)
model += (start + df_data['duration'] == end)

# No two active alternatives on the same machine may overlap; (ab)use cumulative with 'active' as demand.
for mach_id, group in df_data.groupby('machine_id'):
    sel = group.index
    model += cp.Cumulative(start[sel], group['duration'].values, end[sel], active[sel], capacity=1)

# --- Objectives ---
# Makespan: max over all active alternatives
makespan = cp.intvar(0, horizon, name="makespan")
for i in range(num_alternatives):
    model += active[i].implies(makespan >= end[i])  # end times of actives determines makespan

# Total energy consumption
total_energy = cp.sum(df_data['energy'] * active)

# Minimize makespan first, then total energy
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
    px.timeline(df_solution, x_start="start", x_end="end", y="machine_id", color="job_id", text="energy").show()
else:
    print("No solution found.")


def compare_solvers(model):
    """
    Compare the runtime of all installed solvers on the given model.
    """
    print("Solving with all installed solvers...")
    for solvername in cp.SolverLookup.solvernames():
        try:
            model.solve(solver=solvername, time_limit=10)  # max 10 seconds
            print(f"{solvername}: {model.status()}")
        except Exception as e:
            print(f"{solvername}: Not run -- {str(e)}")

# --- bonus: compare the runtime of all installed solvers ---
# compare_solvers(model)
