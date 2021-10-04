#!/usr/bin/python3
"""
Packing problem in CPMPy + Visualization with Pillow

Reworked based on Alexander Schiendorfer's
https://github.com/Alexander-Schiendorfer/cp-examples/tree/main/packing

Given n squares with the i-th square having side i, the program finds the (minimum) 2D area to pack these squares.
"""

from cpmpy import *
from PIL import Image, ImageDraw, ImageFont

# Number of squares we want to pack
n = 8

# Dimension bounds of the overall needed space
max_side = sum([i for i in range(1, n+1)])
min_area = sum([(i*i) for i in range(1, n+1)])

# Decision variables
rect_height = intvar(n, max_side)
rect_width = intvar(n, max_side)
rect_area = intvar(min_area, n*max_side)
x = intvar(0, max_side, shape = n)
y = intvar(0, max_side, shape = n)

model = Model(
    ### Necessary constraints
    (rect_area == rect_height * rect_width), # Definition of the area of the needed space
    [x[i] + i + 1 <= rect_width for i in range(n)], # Every item has to be within the width of the overall area
    [y[i] + i + 1  <= rect_height for i in range(n)], # Every item has to be withing the height of the overall area
    [(x[i] + i + 1 <= x[j]) | (x[j] + j + 1 <= x[i]) | (y[i] + i + 1 <= y[j]) | (y[j] + j + 1 <= y[i]) for i in range(n) for j in range(n) if i != j], # Every item has to be fully above, below or next to every other item

    ### Additional constraints
    # (rect_width > rect_height),
    # (rect_width >= 10),
    # (rect_height >= 10)
)

# Minimize wrt the overall area
model.minimize(rect_area)

if model.solve():
    print(rect_height.value())
    print(rect_width.value())
    print(rect_area.value())
    print(x.value())
    print(y.value())

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
    myFont = ImageFont.truetype("arialbd.ttf", 20)

    # Draw overall rectangle
    shape = [(start_x, start_y), (start_x + rect_width.value() * pixel_unit, start_y + rect_height.value() * pixel_unit)]
    img1.rectangle(shape, fill ="#000000", outline ="white")
    # Draw squares
    for i in range(n):
        shape_x, shape_y = x[i].value(), y[i].value()
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
    msg =  f"Area is {rect_area.value()}, width = {rect_width.value()}, height = {rect_height.value()}"
    w, h = img1.textsize(msg, font=myFont)
    img1.text((center_x - w / 2, center_y - h / 2), msg, fill="white", font=myFont)

    img.show()