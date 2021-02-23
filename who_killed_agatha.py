#!/usr/bin/python3
"""
Who killed agatha? problem in CPmpy

Based on the Numberjack model of Hakan Kjellerstrand
see also: http://www.hakank.org/constraint_programming_blog/2014/11/decision_management_community_november_2014_challenge_who_killed_agath.html
"""
from cpmpy import *
from enum import Enum
import numpy

# Agatha, the butler, and Charles live in Dreadsbury Mansion, and 
# are the only ones to live there. 
n = 3
(agatha, butler, charles) = range(n) # enum constants
    
# Who killed agatha?
victim = agatha
killer = IntVar(0,2)

constraint = []
# A killer always hates, and is no richer than his victim. 
hates = BoolVar((n,n))
constraint += [ hates[killer, victim] == 1 ]

richer = BoolVar((n,n))
constraint += [ richer[killer, victim] == 0 ]

# implied richness: no one richer than himself, and anti-reflexive
constraint += [ richer[i,i] == 0 for i in range(n) ]
constraint += [ (richer[i,j] == 1) == (richer[j,i] == 0) for i in range(n) for j in range(n) if i != j ]

# Charles hates noone that Agatha hates. 
constraint += [ implies(hates[agatha,i] == 1, hates[charles,i] == 0) for i in range(n) ]

# Agatha hates everybody except the butler. 
#cons_aga = (hates[agatha,(agatha,charles,butler] == [1,1,0])
constraint += [ hates[agatha,agatha]  == 1,
                hates[agatha,charles] == 1,
                hates[agatha,butler]  == 0 ]

# The butler hates everyone not richer than Aunt Agatha. 
constraint += [ implies(richer[i,agatha] == 0, hates[butler,i] == 1) for i in range(n) ]

# The butler hates everyone whom Agatha hates. 
constraint += [ implies(hates[agatha,i] == 1, hates[butler,i] == 1) for i in range(n) ]

# Noone hates everyone. 
constraint += [ sum([hates[i,j] for j in range(n)]) <= 2 for i in range(n) ]
     

model = Model(constraint)
stats = model.solve()

#print(model)
print("killer ID:",killer.value())
