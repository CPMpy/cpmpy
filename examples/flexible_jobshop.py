# Parallel machine scheduling: a set of jobs must be scheduled, each can be run on compatible machines,
# with different duration and energy consumption. Minimize makespan and total energy consumption
import cpmpy as cp
import pandas as pd
import random; random.seed(1)
SHOW_VISUALISATION = False

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
    if SHOW_VISUALISATION:
        import plotly.express as px
        df_solution = df_data[active.value() == True].copy()  # Select rows where active is True
        df_solution["start"] = pd.to_datetime(start[df_solution.index].value(), unit="m")
        df_solution["end"] = pd.to_datetime(end[df_solution.index].value(), unit="m")
        import plotly.io as pio; pio.renderers.default = "browser"
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