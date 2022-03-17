#!/usr/bin/python3
"""
Packing problem in CPMPy + Visualization with Pillow

Reworked based on Alexander Schiendorfer's
https://github.com/Alexander-Schiendorfer/cp-examples/tree/main/packing

Given n squares with the i-th square having side i, the program finds the (minimum) 2D area to pack these squares.
"""

from cpmpy import *
from PIL import Image, ImageDraw, ImageFont # pip install pillow

def run():
    # Number of squares we want to pack
    n = 8

    (model, vars) = model_sq_packing(n)

    if model.solve():
        for (name, var) in vars.items():
            print(f"{name}:\n{var.value()}")
        visualize_sq_packing(vars, n)

def model_sq_packing(n):
    # Dimension bounds of the overall needed space
    max_side = sum([i for i in range(1, n+1)])
    min_area = sum([(i*i) for i in range(1, n+1)])

    # Decision variables
    rect_height = intvar(n, max_side)
    rect_width = intvar(n, max_side)
    rect_area = intvar(min_area, n*max_side)
    x = intvar(0, max_side, shape = n)
    y = intvar(0, max_side, shape = n)

    m = Model()
    ### Necessary constraints
    # Definition of the area of the needed space
    m += (rect_area == rect_height * rect_width)

    # Every item has to be within the width of the overall area
    m += [x[i] + i + 1 <= rect_width for i in range(n)]

    # Every item has to be withing the height of the overall area
    m += [y[i] + i + 1  <= rect_height for i in range(n)]

    # Every item has to be fully above, below or next to every other item
    for i in range(n): 
        for j in range(n): 
            if i != j: 
                m += ((x[i] + i + 1 <= x[j]) | (x[j] + j + 1 <= x[i]) | (y[i] + i + 1 <= y[j]) | (y[j] + j + 1 <= y[i]))

    # Minimize wrt the overall area
    m.minimize(rect_area)

    return (m, {"rect_height": rect_height, "rect_width": rect_width, "rect_area": rect_area, "x": x, "y": y})

# The remaining code below is exclusively focused on the visualization of the solution
def visualize_sq_packing(vars, n):
    # Extract separate decision variables
    rect_width = vars["rect_width"].value()
    rect_height = vars["rect_height"].value()
    rect_area = vars["rect_area"].value()
    x = vars["x"].value()
    y = vars["y"].value()

    # Draw solution
    # Define start location image & unit size
    start_x, start_y = 20, 40
    pixel_unit = 50

    imwidth, imheight = rect_width * pixel_unit + 2 * start_x, start_y * 2 + rect_height * pixel_unit

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
    shape = [(start_x, start_y), (start_x + rect_width * pixel_unit, start_y + rect_height * pixel_unit)]
    img1.rectangle(shape, fill ="#000000", outline ="white")
    # Draw squares
    for i in range(n):
        shape_x, shape_y = x[i], y[i]
        # Item coordinates
        draw_start_x, draw_start_y = start_x + pixel_unit *  shape_x, start_y + pixel_unit * shape_y
        draw_end_x, draw_end_y = draw_start_x + (i+1) * pixel_unit, draw_start_y + (i+1) * pixel_unit
        
        # Draw item rectangle
        shape = [(draw_start_x, draw_start_y), (draw_end_x, draw_end_y)]
        img1.rectangle(shape, fill ="#005A9B", outline ="white")
        # Add item dimension
        msg =  f"{(i+1)} x {(i+1)}"
        w, h = img1.textsize(msg, font=myFont)
        center_x, center_y = (draw_start_x + draw_end_x) / 2, (draw_start_y + draw_end_y) / 2
        img1.text((center_x - w / 2, center_y - h / 2), msg, fill="white", font=myFont)

    # Show solution data
    center_x, center_y = imwidth / 2, start_y / 2
    msg =  f"Area is {rect_area}, width = {rect_width}, height = {rect_height}"
    w, h = img1.textsize(msg, font=myFont)
    img1.text((center_x - w / 2, center_y - h / 2), msg, fill="white", font=myFont)

    img.show()

if __name__ == "__main__":
    run()