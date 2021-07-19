#!/usr/bin/python3
"""
The 'frietkot' problem; invented by Tias to explain SAT and SAT solving
http://homepages.vub.ac.be/~tiasguns/frietkot/
"""
from cpmpy import *

# Construct the model.
(mayo, ketchup, curry, andalouse, samurai) = boolvar(5)

# Pure CNF
Nora = mayo | ketchup
Leander = ~samurai | mayo
Benjamin = ~andalouse | ~curry | ~samurai
Behrouz = ketchup | curry | andalouse
Guy = ~ketchup | curry | andalouse
Daan = ~ketchup | ~curry | andalouse
Celine = ~samurai
Anton = mayo | ~curry | ~andalouse
Danny = ~mayo | ketchup | andalouse | samurai
Luc = ~mayo | samurai

allwishes = [Nora, Leander, Benjamin, Behrouz, Guy, Daan, Celine, Anton, Danny, Luc]

model = Model(allwishes)
if model.solve():
    print("Mayonaise = ", mayo.value())
    print("Ketchup = ", ketchup.value())
    print("Curry Ketchup = ", curry.value())
    print("Andalouse = ", andalouse.value())
    print("Samurai = ", samurai.value())
else:
    print("No solution found")
