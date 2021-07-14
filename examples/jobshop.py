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

jobs_data = [  # task = (machine_id: duration_time).
        {0:3,1:2,2:2},#job0 
        {0:2,2:1,1:4},# job1
        {1:4,2:3} #job2
    ]
machines_count = 1 + max([mach for job in jobs_data for mach in job.keys()])
max_duration = sum([dur for job in jobs_data for dur in job.values()])

jobs_count = len(jobs_data)
all_jobs = range(jobs_count)
all_machines = range(machines_count)


start_time = intvar(0, max_duration, shape=(machines_count,jobs_count))
end_time = intvar(0, max_duration, shape=(machines_count,jobs_count))

model = Model()
# start + dur = end
for j in all_jobs:
    for (m,dur) in jobs_data[j].items():
        model += (start_time[m,j] + dur == end_time[m,j])

# Precedence constraint per job
for j in all_jobs:
    for m1,m2 in combinations(jobs_data[j].keys(),2):
            model += (end_time[m1,j] <= start_time[m2,j])

# No overlap constraint: one starts before other one ends
for m in all_machines:
    for j1,j2 in (combinations(all_jobs, 2)):
        model += (start_time[m,j1] >= end_time[m,j2]) | \
                 (start_time[m,j2] >= end_time[m,j1])

makespan = Maximum(end_time)

model.minimize(makespan)
val = model.solve()
#print(model.status())

print("Makespan:",makespan.value())
print("Schedule:")
grid = -8*np.ones((machines_count, makespan.value()), dtype=int)
for j in all_jobs:
    for m in jobs_data[j]:
        grid[m,start_time[m,j].value():end_time[m,j].value()] = j
        #print(f"Job {j} starts at machine {m} at {start_time[m,j].value()} and ends at {end_time[m,j].value()}")
print(grid)
