#!/usr/bin/python3
import numpy as np
from itertools import combinations
from cpmpy import *
"""
taken from https://developers.google.com/optimization/scheduling/job_shop
each task is labeled by a pair of numbers (m, p) 
where m is the number of the machine the task must be processed on and
p is the processing time of the task

In this example, we have three jobs
job 0 = [(0, 3), (1, 2), (2, 2)]
job 1 = [(0, 2), (2, 1), (1, 4)]
job 2 = [(1, 4), (2, 3)]
For example, job 0 has three tasks. 
The first, (0, 3), must be processed on machine 0 in 3 units of time

There are two types of constraints for the job shop problem
1. Precedence constraint:  for any two consecutive tasks in the same job, 
the first must be completed before the second can be started.
2. No overlap constraints: a machine can't work on two tasks at the same time
The objective of the job shop problem is to minimize the makespan: 
the length of time from the earliest start time of the jobs to the latest end time.
"""

jobs_data = cpm_array([ # (job, machine) = duration
    [3,2,2], # job 0
    [2,1,4], # job 1
    [0,4,3], # job 2 (duration 0 = not used)
])
max_duration = sum(jobs_data.flat) # TODO, fixme

jobs_count, machines_count = jobs_data.shape
start_time = intvar(0, max_duration, shape=(machines_count,jobs_count))
end_time = intvar(0, max_duration, shape=(machines_count,jobs_count))

all_jobs = range(jobs_count)
all_machines = range(machines_count)

model = Model()
# start + dur = end
model += (start_time + jobs_data.T == end_time)

# Precedence constraint per job
for m1,m2 in combinations(all_machines,2): # [(0,1), (0,2), (1,2)]
    model += (end_time[m1,:] <= start_time[m2,:])

# No overlap constraint: one starts before other one ends
for j1,j2 in combinations(all_jobs, 2):
    model += (start_time[:,j1] >= end_time[:,j2]) | \
             (start_time[:,j2] >= end_time[:,j1])

makespan = max(end_time)
model.minimize(makespan)

#print(model.status())
val = model.solve()
print("Makespan:",makespan.value())
print("Schedule:")
grid = -8*np.ones((machines_count, makespan.value()), dtype=int)
for j in all_jobs:
    for m in all_machines:
        grid[m,start_time[m,j].value():end_time[m,j].value()] = j
print(grid)
