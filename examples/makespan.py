#!/usr/bin/python3
from cpmpy import *
import numpy as np
from itertools import combinations
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
jobs_data = [  # task = (machine_id: processing_time).
        {0:3,1:2,2:2},#job0 
        {0:2,2:1,1:4},# job1
        {1:4,2:3} #job2
    ]

jobs_count = len(jobs_data)
all_jobs = range(jobs_count)
machines_count = 1 + max([task for job in jobs_data for task in job])
all_machines = range(machines_count)
max_duration = sum([job[task] for job in jobs_data for task in job])


makespan = IntVar(0,max_duration)
start_time = IntVar(0, max_duration, shape=(machines_count,jobs_count))
end_time = IntVar(0, max_duration, shape=(machines_count,jobs_count))

constraint  = []
for j in all_jobs:
    for m in all_machines:
        if m in jobs_data[j]:
            constraint += [start_time[m,j]+jobs_data[j][m] == end_time[m,j] ]
# Precedence constraint
for j in all_jobs:
    for m1,m2 in combinations(jobs_data[j],2):
            # print(j,m1,m2)
            constraint += [(end_time[m1,j] <= start_time[m2,j])]


# No overlap constraint
for m in all_machines:
    for j1,j2 in (combinations(all_jobs, 2)):
        # print(m,j1,j2)
        constraint += [any([start_time[m,j1] >= end_time[m,j2] , 
        start_time[m,j2] >= end_time[m,j1] ])]

for j in all_jobs:
    for m in all_machines:
        constraint += [makespan >= end_time[m,j]]
model = Model(constraint, minimize=makespan)
stats = model.solve()
print(stats)
print("Optimal Schedule Length: ",makespan.value())
start_solution = start_time.value()
end_solution = end_time.value()
print("Optimal Scheduling")
for j in all_jobs:
    for m in all_machines:
        # if m in jobs_data[j]:
            print("Job {} starts at machine {} at {} and ends at {}".format(j,m,start_solution[m,j],end_solution[m,j]))
            