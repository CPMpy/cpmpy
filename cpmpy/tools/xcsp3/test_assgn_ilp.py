import cpmpy as cp
import xcsp3_globals as xglobals
import numpy as np

# Sample data
n_workers = 3
n_tasks = 4

# Value matrix: value[i,j] represents the value of worker i doing task j
value = np.array([
    [5, 2, 3, 4],
    [3, 4, 5, 2],
    [2, 5, 4, 3],
])

# Create the model
model = cp.Model()

# Decision variables: task[i] represents which task is assigned to worker i
task = cp.intvar(0, n_tasks-1, shape=n_workers, name="task")

# Each worker works on a different task
model += cp.AllDifferent(task)
model += cp.sum(cp.Count(task, 0), cp.Count(task, 2)) <= 3  # to test count

# Objective: maximize sum of values
obj = cp.intvar(np.min(value), np.sum(value), name="obj")
model += (obj == sum(xglobals.Element(value[i], task[i]) for i in range(n_workers)))
model.maximize(obj)

print("CPMpy model:")
print(model)

print("---------------------------")
print("Ortools transformed model:")
print(cp.SolverLookup.get("ortools").transform(model.constraints))
print(model.solve(), model.status(), obj.value())

print("---------------------------")
print("Exact transformed model:")
print(cp.SolverLookup.get("exact").transform(model.constraints))
obj._value = None
m2 = cp.Model(cp.SolverLookup.get("exact").transform(model.constraints),
              maximize=obj)
print(m2.solve(), m2.status(), obj.value())

