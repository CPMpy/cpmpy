"""
# Based on PyCSP3's Zebra.py by C. Lecoutre
# https://github.com/xcsp3team/pycsp3/blob/master/problems/csp/single/Zebra.py

The Zebra puzzle (sometimes referred to as Einstein's puzzle) is defined as follows.
There are five houses in a row, numbered from left to right.
Each of the five houses is painted a different color, and has one inhabitant.
The inhabitants are all of different nationalities, own different pets, drink different beverages and have different jobs.
We know that:
 - colors are yellow, green, red, white, and blue
 - nations of inhabitants are italy, spain, japan, england, and norway
 - pets are cat, zebra, bear, snails, and horse
 - drinks are milk, water, tea, coffee, and juice
 - jobs are painter, sculptor, diplomat, pianist, and doctor
 - the painter owns the horse
 - the diplomat drinks coffee
 - the one who drinks milk lives in the white house
 - the Spaniard is a painter
 - the Englishman lives in the red house
 - the snails are owned by the sculptor
 - the green house is on the left of the red one
 - the Norwegian lives on the right of the blue house
 - the doctor drinks milk
 - the diplomat is Japanese
 - the Norwegian owns the zebra
 - the green house is next to the white one
 - the horse is owned by the neighbor of the diplomat
 - the Italian either lives in the red, white or green house
"""

from cpmpy import *

n_houses = 5

# colors[i] is the house of the ith color
yellow, green, red, white, blue = colors = intvar(0,n_houses-1, shape=n_houses)

# nations[i] is the house of the inhabitant with the ith nationality
italy, spain, japan, england, norway = nations = intvar(0,n_houses-1, shape=n_houses)

# jobs[i] is the house of the inhabitant with the ith job
painter, sculptor, diplomat, pianist, doctor = jobs = intvar(0,n_houses-1, shape=n_houses)

# pets[i] is the house of the inhabitant with the ith pet
cat, zebra, bear, snails, horse = pets = intvar(0,n_houses-1, shape=n_houses)

# drinks[i] is the house of the inhabitant with the ith preferred drink
milk, water, tea, coffee, juice = drinks = intvar(0,n_houses-1, shape=n_houses)

m = Model(
    AllDifferent(colors),
    AllDifferent(nations),
    AllDifferent(jobs),
    AllDifferent(pets),
    AllDifferent(drinks),

    painter == horse,
    diplomat == coffee,
    white == milk,
    spain == painter,
    england == red,
    snails == sculptor,
    green + 1 == red,
    blue + 1 == norway,
    doctor == milk,
    japan == diplomat,
    norway == zebra,
    abs(green - white) == 1,
    #horse in {diplomat - 1, diplomat + 1},
    (horse == diplomat-1)|(horse == diplomat+1),
    #italy in {red, white, green}
    (italy == red)|(italy == white)|(italy == green),
)
m.solve()
print("The zebra (owner) lives in house number",zebra.value())
