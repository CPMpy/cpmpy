#!/usr/bin/python3
"""
Scheduling problem in CPMPy + Visualization with Pillow

Reworked based on Alexander Schiendorfer's
https://github.com/Alexander-Schiendorfer/cp-examples/tree/main/scheduling

Given some jobs with corresponding tasks performed on multiple machines, 
the program finds a schedule that satisfies all priority constraints while minimizing the overall timespan.
"""

from cpmpy import *
from PIL import Image, ImageDraw, ImageFont # pip install pillow

def run():
    # All data related to the scheduling
    jobs = ['A', 'B', 'C', 'D'] # Different jobs
    lastT = 20 # Time limit
    dur = cpm_array([[5, 2, 3 ], [4, 5, 1], [ 3, 4, 2 ], [1, 1, 1]]) # Amount of time needed per task
    taskToMach = cpm_array([[1, 2, 3 ],[ 2, 1, 3],[ 2, 3, 1 ], [3, 2, 1]]) # On what machine each task has to be performed

    (model, vars) = model_scheduling(jobs, lastT, dur, taskToMach)

    if model.solve():
        for (name, var) in vars.items():
            print(f"{name}:\n{var.value()}")
        visualize_scheduling(vars, lastT, jobs, taskToMach)

def model_scheduling(jobs, lastT, dur, taskToMach):
    nTasks = len(dur[0]) # Amount of tasks per job
    nJobs = len(jobs) # Amount of jobs

    # Decision variables
    start = intvar(0, lastT, shape=(nJobs, nTasks)) # Start time of each task
    end = intvar(0, lastT, shape=(nJobs, nTasks)) # End time of each task
    makespan = intvar(0, lastT) # Overall needed time

    m = Model()
    # Necessary constraints
    # The end of every task is the sum of its start and duration
    m += [end[i,j] == start[i,j] + dur[i,j] for i in range(nJobs) for j in range(nTasks)] 

    # In every pair of jobs on the same machine, one comes before the other
    for i1 in range(nJobs):
        for i2 in range(nJobs): 
            for j1 in range(nTasks): 
                for j2 in range(nTasks):
                    if (i1 != i2 and taskToMach[i1,j1] == taskToMach[i2,j2]):
                        m += ((end[i1,j1] <= start[i2,j2]) | (end[i2,j2] <= start[i1,j1]))
    
    # Within a job, tasks have a fixed order
    for i in range(nJobs): 
        for j in range(nTasks-1):
            m += (end[i,j] <= start[i,j+1])

    # The makespan is defined as the total needed time to finish all jobs
    m += (makespan == max([end[i,j] for i in range(len(jobs)) for j in range(nTasks)]))

    # Optional constraints
    # The 2nd task of job B has to come before all 2nd tasks of other jobs
    for i in range(nJobs):
        if i != 1:
            m += (start[i,1] >= start[1,1])
    

    # Minimize wrt the makespan
    m.minimize(makespan)

    return (m, {"start": start, "end": end, "makespan": makespan})

# The remaining code below is exclusively focused on the visualization of the solution
def visualize_scheduling(vars, lastT, jobs, taskToMach):
    nTasks = len(taskToMach[0]) # Amount of tasks per job
    nMachines = max([taskToMach[i,j] for i in range(len(taskToMach)) for j in range(len(taskToMach[0]))]) # Amount of machines present

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
    # To use Arial instead of the default Pillow font on Debian-based Linux distributions, run the following in terminal:
    # 'sudo apt install ttf-mscorefonts-installer'
    # 'sudo fc-cache -f'

    try:
        myFont = ImageFont.truetype("arialbd.ttf", 20)
    except:
        myFont = ImageFont.load_default()

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
            on_machine = taskToMach[i,t] - 1
            start_m_x, start_m_y = machine_upper_lefts[on_machine]

            start_rect_x, start_rect_y = start_m_x + start[i,t].value() * pixel_unit, start_m_y + inner_sep
            end_rect_x, end_rect_y = start_m_x + end[i,t].value() * pixel_unit, start_m_y + pixel_task_height - inner_sep

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