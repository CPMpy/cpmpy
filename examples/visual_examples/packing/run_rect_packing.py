#!/usr/bin/python3
"""
Packing problem in CPMPy + Visualization with Pillow

Reworked based on Alexander Schiendorfer's
https://github.com/Alexander-Schiendorfer/cp-examples/tree/main/packing

Given some rectangular 2D items, the program finds the (minimum) 2D area to pack these items.
"""

import builtins
from cpmpy import *
import numpy as np
from PIL import Image, ImageDraw, ImageFont # pip install pillow

def run():
    # 2D dimensions of the items to be packed
    widths  = cpm_array([5, 6, 4, 3, 2, 4, 3, 1, 2, 1, 7, 3])
    heights = cpm_array([1, 2, 3, 2, 1, 2, 4, 6, 5, 1, 1, 2])

    (model, vars) = model_rect_packing(widths, heights)

    if model.solve():
        for (name, var) in vars.items():
            print(f"{name}:\n{var.value()}")
        visualize_rect_packing(vars, widths, heights)

def model_rect_packing(widths, heights):
    # Number of different items
    n = len(widths)

    # Max dimensions of the overall needed space
    max_width_rect = sum(widths)
    max_height_rect = sum(heights)

    # Min dimensions of the overall needed space
    min_width_rect = builtins.max(widths)
    min_height_rect = builtins.max(heights)
    
    # Decision variables
    rect_height = intvar(min_height_rect, max_height_rect)
    rect_width = intvar(min_width_rect, max_width_rect)

    x = intvar(0, max_width_rect, shape = n)
    y = intvar(0, max_height_rect, shape = n)

    m = Model()

    ### Necessary constraints
    # Every item has to be within the width of the overall area
    m += (x + widths <= rect_width)

    # Every item has to be within the height of the overall area
    m += (y + heights <= rect_height)
    
    # Every item has to be fully above, below or next to every other item
    for i in range(n):
        for j in range(n):
            if i != j:
                m += ((x[i] + widths[i] <= x[j]) | (x[j] + widths[j] <= x[i]) | (y[i] + heights[i] <= y[j]) | (y[j] + heights[j] <= y[i]))

    ### Optional constraints
    # The needed space needs to be wider than taller
    # (rect_width > rect_height),

    # The needed space has to have a width larger than 10
    # (rect_width >= 10),

    # The needed space has to have a height larger than 10
    # (rect_height >= 10)

    # Minimize wrt the overall area
    m.minimize((rect_height*rect_width))

    return (m, {"rect_height": rect_height, "rect_width": rect_width, "rect_area": rect_height*rect_width, "x": x, "y": y})

# The remaining code below is exclusively focused on the visualization of the solution
def visualize_rect_packing(vars, widths, heights):
    n = len(widths) # Amount of items

    rect_width = vars["rect_width"]
    rect_height = vars["rect_height"]
    rect_area = vars["rect_area"]
    x = vars["x"]
    y = vars["y"]

    # Draw solution
    # Define start location image & unit size
    start_x, start_y = 20, 40
    pixel_unit = 50

    imwidth, imheight = rect_width.value() * pixel_unit + 2 * start_x, start_y * 2 + rect_height.value() * pixel_unit

    # Create new Image object
    img = Image.new("RGB", (imwidth, imheight))

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

    # Draw overall rectangle
    shape = [(start_x, start_y), (start_x + rect_width.value() * pixel_unit, start_y + rect_height.value() * pixel_unit)]
    img1.rectangle(shape, fill ="#000000", outline ="white")
    # Draw items
    for i in range(n):
        shape_x, shape_y = x[i].value(), y[i].value()
        # Item coordinates
        draw_start_x, draw_start_y = start_x + pixel_unit *  shape_x, start_y + pixel_unit * shape_y
        draw_end_x, draw_end_y = draw_start_x + widths[i] * pixel_unit, draw_start_y + heights[i] * pixel_unit
        
        # Draw item rectangle
        shape = [(draw_start_x, draw_start_y), (draw_end_x, draw_end_y)]
        img1.rectangle(shape, fill ="#005A9B", outline ="white")
        # Add item dimension
        msg =  f"{widths[i]} x {heights[i]}"
        w, h = img1.textsize(msg, font=myFont)
        center_x, center_y = (draw_start_x + draw_end_x) / 2, (draw_start_y + draw_end_y) / 2
        img1.text((center_x - w / 2, center_y - h / 2), msg, fill="white", font=myFont)

    # Show solution data
    center_x, center_y = imwidth / 2, start_y / 2
    msg =  f"Area is {rect_area.value()}, width = {rect_width.value()}, height = {rect_height.value()}"
    w, h = img1.textsize(msg, font=myFont)
    img1.text((center_x - w / 2, center_y - h / 2), msg, fill="white", font=myFont)

    img.show()

if __name__ == "__main__":
    run()