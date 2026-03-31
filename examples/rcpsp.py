#!/usr/bin/env python
from cpmpy import *

# toy model taken from :
# https://python-mip.readthedocs.io/en/latest/examples.html#resource-constrained-project-scheduling
# ported to CPMpy by Guillaume Poveda

# Data
durations = cpm_array([0, 3, 2, 5, 4, 2, 3, 4, 2, 4, 6, 0])

resource_needs = cpm_array([[0, 0], [5, 1], [0, 4], [1, 4], [1, 3], [3, 2], [3, 1], [2, 4], [4, 0], [5, 2], [2, 5], [0, 0]])

resource_capacities = cpm_array([6, 8])

successors_link = cpm_array([[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 9], [2, 10], [3, 8], [4, 6], [4, 7], [5, 9], [5, 10], [6, 8], [6, 9], [7, 8], [8, 11], [9, 11], [10, 11]])

nb_resource = len(resource_capacities)
nb_jobs = len(durations)
max_duration = sum(durations)  # dummy upper bound, can be improved of course

# Variables
start_time = intvar(0, max_duration, shape=nb_jobs)


model = Model()
# Precedence constraints
for j in range(successors_link.shape[0]):
    model += start_time[successors_link[j, 1]] >= start_time[successors_link[j, 0]]+durations[successors_link[j, 0]]

# Cumulative resource constraint
for r in range(nb_resource):
    model += Cumulative(start=start_time, duration=durations, end=start_time+durations,
                        demand=resource_needs[:, r], capacity=resource_capacities[r])

makespan = max(start_time)
model.minimize(makespan)

model.solve(solver="ortools")
print("Start times:", start_time.value())


def check_solution(start_time_values):
    for j in range(successors_link.shape[0]):
        assert start_time_values[successors_link[j, 1]] >= start_time_values[successors_link[j, 0]]+\
               durations[successors_link[j, 0]]
    for t in range(max(start_time_values)+1):
        active_index = [i for i in range(nb_jobs) if durations[i] > 0 and
                        start_time_values[i] <= t < start_time_values[i]+durations[i]]
        for r in range(nb_resource):
            consumption = sum([resource_needs[i, r] for i in active_index])
            if consumption>resource_capacities[r]:
                print(t, r, consumption, resource_capacities[r])
            assert consumption <= resource_capacities[r]


check_solution(start_time.value())
print("Solution passed all checks.")
