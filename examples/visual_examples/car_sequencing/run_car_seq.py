#!/usr/bin/python3
"""
Sequencing problem in CPMPy + Visualization with Pillow

Reworked based on Alexander Schiendorfer's
https://github.com/Alexander-Schiendorfer/cp-examples/tree/main/car-sequencing

Given different kinds of cars, an overall car demand and some constraints on the sequencing wrt to the cars' properties,
the program finds a feasible ordering of cars in a time schedule.
"""

import builtins
from cpmpy import *
from PIL import Image, ImageDraw, ImageFont
import datetime

def run():
    # All data related to the sequencing
    nOptions = 5
    nCarConfigs = 6
    at_most = [1, 2, 2, 2, 1]
    per_slots = [2, 3, 3, 5, 5]
    demand = [1,1,2,2,2,2]
    nCars = sum(demand)
    nSlots = nCars
    requires = [
    [1, 0, 1, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 0, 0, 1],
    [0, 1, 0, 1, 0],
    [1, 0, 1, 0, 0],
    [1, 1, 0, 0, 0]]

    (model, vars) = model_sequence(nCarConfigs, nSlots, nOptions, demand, per_slots, at_most, requires)

    if model.solve():
        for v in vars:
            print(v + ":\n" + str(vars[v].value()))
        visualize_sequence(vars, nSlots, nOptions, nCarConfigs, requires, demand, at_most, nCars, per_slots)

def model_sequence(nCarConfigs, nSlots, nOptions, demand, per_slots, at_most, requires):
    # Decision variables
    line = intvar(1, nCarConfigs, shape = nSlots)
    setup = boolvar(shape = (nSlots, nOptions))

    model = Model(
        [sum((x == (i+1)) for x in line) == demand[i] for i in range(nCarConfigs)] # The amount of each type of car in the timetable has to be equal to the demand for that type
    )

    for o in range(nOptions):
        model += [sum(setup[s2][o] == True for s2 in range(s, s + per_slots[o])) <= at_most[o] for s in range(nSlots - per_slots[o])] # Check that no more than "at most" car properties are used per "per_slots" slots

    requires = cpm_array(requires) # So it can be indexed by a variable
    model += [setup[s,o] == requires[line[s]-1,o] for s in range(nSlots) for o in range(nOptions)] # Make sure that the properties in the timetable correspond to those per car type

    return (model, {"line": line, "setup": setup})

def visualize_sequence(vars, nSlots, nOptions, nCarConfigs, requires, demand, at_most, nCars, per_slots):
    def writeAt(x, y, msg, draw, myFont, fillc):
        # Write a label there
        text_w, text_h = draw.textsize(msg, font=myFont)
        draw.text((x - text_w / 2, y - text_h / 2), msg, fill=fillc, font=myFont)

    def drawOptionHeader(start_x, start_y):
        for i in range(nOptions):
            start_opt_x, start_opt_y = start_x + i * opt_width, start_y
            end_opt_x, end_opt_y = start_opt_x + opt_width, start_opt_y + line_height
            draw.rectangle((start_opt_x, start_opt_y, end_opt_x, end_opt_y), fill=header_gray, outline="black")
            writeAt(start_opt_x + opt_width/2, start_opt_y + line_height/2, str(i+1), draw, myFont, "black")
    
    def writtenRectangle(x, y, w, h, msg, fillc, fontc):
            draw.rectangle((x, y, x + w, y + h), fill=fillc, outline="black")
            writeAt(x + w / 2, y + h / 2, msg, draw, myFont, fontc)

    def drawOptions(start_x, start_y, opts):
        for i in range(nOptions):
            start_opt_x, start_opt_y = start_x + i * opt_width, start_y
            end_opt_x, end_opt_y = start_opt_x + opt_width, start_opt_y + line_height
            fillc = emph_cs if opts[i] > 0 else "white"
            draw.rectangle((start_opt_x, start_opt_y, end_opt_x, end_opt_y), fill=fillc, outline="black")

    def writeCapacityLine(start_cap_x, start_cap_y):
        writtenRectangle(start_cap_x, start_cap_y, car_conf_width, line_height, "Capacity", \
                        fillc=header_gray, fontc="black")
        for i in range(nOptions):
            start_opt_x, start_opt_y = start_cap_x + car_conf_width + i * opt_width, start_cap_y
            end_opt_x, end_opt_y = start_opt_x + opt_width, start_opt_y + line_height
            draw.rectangle((start_opt_x, start_opt_y, end_opt_x, end_opt_y), fill=header_gray, outline="black")
            writeAt(start_opt_x + opt_width / 2, start_opt_y + line_height / 2,\
                    f"{at_most[i]}/{per_slots[i]}", draw, myFont, "black")

    line = vars["line"]

    # Datetime generators for printing in visualisation
    datetime_object = datetime.datetime.strptime('14:00', '%H:%M')
    date_list = [datetime_object + datetime.timedelta(minutes=15*x) for x in range(0, nSlots)]
    print_dates = [d.strftime('%H:%M') for d in date_list]

    # Define drawing properties
    vert_pad = 50
    hor_pad = 50
    slot_width = 80
    car_conf_width = 200
    demand_width = 100
    opt_width = 50
    line_height = 40

    problem_width = car_conf_width + nOptions*opt_width + demand_width
    problem_height = (nCarConfigs + 2) * line_height

    solution_width = slot_width + car_conf_width + nOptions * opt_width
    solution_height = (nSlots + 1) * line_height

    imwidth = 4*hor_pad + problem_width + solution_width
    imheight = 2*vert_pad + builtins.max(problem_height, solution_height)
    center_x = imwidth/2

    # Create new Image object
    img = Image.new("RGB", (imwidth, imheight), (255, 255,255))

    # Create rectangle image
    draw = ImageDraw.Draw(img)

    # Draw line separating problem and solution
    draw.line((center_x, 0, center_x, imheight), fill="black")
    # Get a font
    myFont = ImageFont.truetype("arialbd.ttf", 20, )

    # Write "problem" and "solution"
    problem_title_x = center_x/2
    problem_title_y = vert_pad / 2
    writeAt(problem_title_x, problem_title_y, "Problem", draw, myFont, "black")

    sol_title_x = center_x + center_x/2
    sol_title_y = vert_pad / 2
    writeAt(sol_title_x, sol_title_y, "Solution", draw, myFont, "black")

    # Write the rectangle surrounding problem and solution
    draw.rectangle((hor_pad, vert_pad, hor_pad+problem_width, vert_pad+problem_height), fill ="white", outline ="black")
    draw.rectangle((center_x + hor_pad, vert_pad, center_x + hor_pad+solution_width, vert_pad+solution_height), fill ="white", outline ="black")

    # Color defs
    header_gray = "#d9d9d9"
    thi_blue = "#04599a"
    config_cs = ["#04599a", "#0683e0", "#53b3fb", "#004200", "#006800", "#009900"]
    font_cs = "#e4e8e4"
    emph_cs ="#ffcc00"

    # Start with the problem header
    problem_start_x, problem_start_y = hor_pad, vert_pad

    drawOptionHeader(problem_start_x + car_conf_width, problem_start_y)
    # Write demand
    start_demand_x, start_demand_y = problem_start_x + car_conf_width + nOptions*opt_width, problem_start_y,
    end_demand_x, end_demand_y =  start_demand_x + demand_width, problem_start_y+line_height
    draw.rectangle((start_demand_x, start_demand_y, end_demand_x, end_demand_y ), fill=thi_blue, outline="black")
    writeAt(start_demand_x + demand_width/2, start_demand_y + line_height/2, "Demand", draw, myFont, "white")

    # For all configs
    for i in range(nCarConfigs):
        start_cc_x, start_cc_y = problem_start_x, problem_start_y + (i + 1) * line_height
        writtenRectangle(start_cc_x, start_cc_y,car_conf_width, line_height, f"Car config {i+1}", \
                        fillc = config_cs[i % len(config_cs)], fontc = font_cs)
        opts = requires[i]
        drawOptions(start_cc_x + car_conf_width, start_cc_y, opts)
        # Draw demand
        writtenRectangle(start_cc_x + car_conf_width + nOptions * opt_width, start_cc_y, demand_width, line_height, \
                        str(demand[i]), fillc="white", fontc=config_cs[i % len(config_cs)])

    # And the capacity line
    start_cap_x, start_cap_y = problem_start_x, problem_start_y + (nCarConfigs + 1) * line_height
    writeCapacityLine(start_cap_x, start_cap_y)
    # The sum of demands
    writtenRectangle(start_cap_x + car_conf_width + nOptions * opt_width, start_cap_y, demand_width, line_height, \
                        str(nCars), fillc=thi_blue, fontc="white")

    # Now the solution header
    sol_start_x, sol_start_y = center_x + hor_pad, vert_pad
    writtenRectangle(sol_start_x, sol_start_y, slot_width, line_height, \
                        "Slots", fillc="white", fontc="BLACK")
    writtenRectangle(sol_start_x + slot_width, sol_start_y, car_conf_width, line_height, \
                        "Car Config (line)", fillc="white", fontc="BLACK")
    drawOptionHeader(sol_start_x + slot_width + car_conf_width, sol_start_y)

    for i in range(nSlots):
        slot_start_x, slot_start_y = sol_start_x, sol_start_y + (i+1) * line_height
        writtenRectangle(slot_start_x, slot_start_y, slot_width, line_height, \
                        print_dates[i], fillc=thi_blue , fontc="white")

        # Which config is placed in slot i?
        c_at_i = line.value()[i]
        writtenRectangle(slot_start_x + slot_width, slot_start_y, car_conf_width, line_height, f"Car config {c_at_i}", \
                        fillc=config_cs[c_at_i % len(config_cs)], fontc=font_cs)
        opts = requires[c_at_i - 1 ]
        drawOptions(slot_start_x + slot_width + car_conf_width, slot_start_y, opts)

    img.show()

if __name__ == "__main__":
    run()