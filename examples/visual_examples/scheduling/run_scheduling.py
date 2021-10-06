#!/usr/bin/python3
"""
Scheduling problem in CPMPy + Visualization with Pillow

Reworked based on Alexander Schiendorfer's
https://github.com/Alexander-Schiendorfer/cp-examples/tree/main/scheduling

Given some jobs with corresponding tasks performed on multiple machines, 
the program finds a schedule that satisfies all priority constraints while minimizing the overall timespan.
"""

from cpmpy import *
from PIL import Image, ImageDraw, ImageFont

def run():
    # All data related to the scheduling
    jobs = ['A', 'B', 'C', 'D']
    lastT = 20
    nTasks = 3
    dur = [ [5, 2, 3 ], [4, 5, 1], [ 3, 4, 2 ], [1, 1, 1]]
    nMachines = 3
    taskToMach = [[1, 2, 3 ],[ 2, 1, 3],[ 2, 3, 1 ], [3, 2, 1]]

    (model, vars) = model_scheduling(jobs, lastT, nTasks, dur, taskToMach)

    if model.solve():
        for v in vars:
            print(v + ":\t" + str(vars[v].value()))
        visualize_scheduling(vars, lastT, nMachines,jobs, nTasks, taskToMach)

def model_scheduling(jobs, lastT, nTasks, dur, taskToMach):

    # Decision variables
    start = intvar(0, lastT, shape=(len(jobs), nTasks))
    end = intvar(0, lastT, shape=(len(jobs), nTasks))
    makespan = intvar(0, lastT)

    model = Model(
        # Necessary constraints
        [end[i][j] == start[i][j] + dur[i][j] for i in range(len(jobs)) for j in range(nTasks)], # The end of every task is the sum of its start and duration
        [(end[i1][j1] <= start[i2][j2]) | (end[i2][j2] <= start[i1][j1]) for i1 in range(len(jobs)) for i2 in range(len(jobs)) for j1 in range(nTasks) for j2 in range(nTasks) if (i1 != i2 and taskToMach[i1][j1] == taskToMach[i2][j2])], # In every pair of jobs on the same machine, one comes before the other
        [end[i][j] <= start[i][j+1] for i in range(len(jobs)) for j in range(nTasks-1)], # Within a job, tasks have a fixed order
        [makespan == max([end[i][j] for i in range(len(jobs)) for j in range(nTasks)])], # The makespan is defined as the total needed time to finish all jobs

        # Optional constraints
        [start[i][1] >= start[1][1] for i in range(len(jobs)) if i != 1] # The 2nd task of job B has to come before all 2nd tasks of other jobs
    )

    # Minimize wrt the makespan
    model.minimize(makespan)

    return (model, {"start": start, "end": end, "makespan": makespan})

def visualize_scheduling(vars, lastT, nMachines,jobs, nTasks, taskToMach):
    makespan = vars["makespan"]
    start = vars["start"]
    end = vars["end"]

    # Draw solution
    # Define start location image & unit sizes
    start_x, start_y = 30, 40
    pixel_unit = 50
    pixel_task_height = 100
    vert_pad = 10

    imwidth, imheight = lastT * pixel_unit + 2 * start_x, start_y + start_x + nMachines * (pixel_task_height + vert_pad)

    # Create new Image object
    img = Image.new("RGB", (imwidth, imheight), (255, 255,255))

    # Create rectangle image
    img1 = ImageDraw.Draw(img)

    # Get a font
    myFont = ImageFont.truetype("arialbd.ttf", 20, )

    # Draw makespan label
    center_x, center_y = imwidth / 2, start_y / 2
    msg =  f"Makespan: {makespan.value()}"
    w, h = img1.textsize(msg, font=myFont)
    img1.text((center_x - w / 2, center_y - h / 2), msg, fill="black", font=myFont)

    task_cs = ["#4bacc6", "#f79646", "#9bbb59"]
    lane_cs = ["#a5d5e2", "#fbcaa2", "#cdddac"]
    lane_border_cs = ["#357d91", "#b66d31", "#71893f"]

    # Draw three rectangles for machines
    machine_upper_lefts = []
    for i in range(nMachines):
        start_m_x, start_m_y = start_x, start_y + i * (pixel_task_height + vert_pad)
        end_m_x, end_m_y = start_m_x + lastT * pixel_unit, start_m_y + pixel_task_height
        machine_upper_lefts += [(start_m_x, start_m_y)]

        shape = [(start_m_x, start_m_y), (end_m_x, end_m_y)]
        img1.rectangle(shape, fill =lane_cs[i], outline =lane_border_cs[i])

    # Draw tasks for each job
    inner_sep = 5
    for i,j in enumerate(jobs):
        job_name = str(j)
        for t in range(nTasks):
            on_machine = taskToMach[i][t] - 1
            start_m_x, start_m_y = machine_upper_lefts[on_machine]

            start_rect_x, start_rect_y = start_m_x + start[i][t].value() * pixel_unit, start_m_y + inner_sep
            end_rect_x, end_rect_y = start_m_x + end[i][t].value() * pixel_unit, start_m_y + pixel_task_height - inner_sep

            shape = [(start_rect_x, start_rect_y), (end_rect_x, end_rect_y)]
            img1.rectangle(shape, fill=task_cs[on_machine], outline=lane_border_cs[on_machine])

            # Write a label for each task of each job
            msg =  f"{job_name}{t+1}"
            text_w, text_h = img1.textsize(msg, font=myFont)
            center_x, center_y = (start_rect_x + end_rect_x) / 2, (start_rect_y + end_rect_y) / 2
            img1.text((center_x - text_w / 2, center_y - text_h / 2), msg, fill="white", font=myFont)

    img.show()

if __name__ == "__main__":
    run()