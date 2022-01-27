#!/usr/bin/python3
"""
Mario problem in CPMpy

Based on the MiniZinc model, same data
"""
import numpy
from cpmpy import *

data = { # a dictionary, json style
  'nbHouses': 15,
  'MarioHouse': 1,
  'LuigiHouse': 2,
  'fuelMax': 2000,
  'goldTotalAmount': 1500,
  'conso': [[0,221,274,808,13,677,670,921,943,969,13,18,217,86,322],[0,0,702,83,813,679,906,246,335,529,719,528,451,242,712],[274,702,0,127,110,72,835,5,161,430,686,664,799,523,73],[808,83,127,0,717,80,31,71,683,668,248,826,916,467,753],[13,813,110,717,0,951,593,579,706,579,101,551,280,414,294],[677,679,72,80,951,0,262,12,138,222,146,571,907,225,938],[670,906,835,31,593,262,0,189,558,27,287,977,226,454,501],[921,246,5,71,579,12,189,0,504,221,483,226,38,314,118],[943,335,161,683,706,138,558,504,0,949,393,721,267,167,420],[969,529,430,668,579,222,27,221,949,0,757,747,980,589,528],[13,719,686,248,101,146,287,483,393,757,0,633,334,492,859],[18,528,664,826,551,571,977,226,721,747,633,0,33,981,375],[217,451,799,916,280,907,226,38,267,980,334,33,0,824,491],[86,242,523,467,414,225,454,314,167,589,492,981,824,0,143],[322,712,73,753,294,938,501,118,420,528,859,375,491,143,0]],
  'goldInHouse': [0,0,40,67,89,50,6,19,47,68,94,86,34,14,14],
}

# Python is offset 0, MiniZinc (source of the data) is offset 1
marioHouse, luigiHouse = data['MarioHouse']-1, data['LuigiHouse']-1 
fuelLimit = data['fuelMax']
nHouses = data['nbHouses']
arc_fuel = data['conso'] # arc_fuel[a,b] = fuel from a to b
arc_fuel = cpm_array(arc_fuel) # needed to do arc_fuel[var1] == var2

# s[i] is the house succeeding to the ith house (s[i]=i if not part of the route)
s = intvar(0,nHouses-1, shape=nHouses, name="s")

model = Model(
    #s should be a path, mimic (sub)circuit by connecting end-point back to start
    s[luigiHouse] == marioHouse,
    Circuit(s),  # should be subcircuit?
)

# consumption, knowing that always conso[i,i]=0 
# node_fuel[i] = arc_fuel[i, successor-of-i]
# observe how we do NOT create auxiliary CP variables here, just a list of expressions...
node_fuel = [arc_fuel[i, s[i]] for i in range(nHouses)]
model += sum(node_fuel) < fuelLimit

# amount of gold earned, only for stops visited, s[i] != i
gold = sum( (s != range(nHouses))*data['goldInHouse'] )
model.maximize(gold)

assert model.solve(), "Model is UNSAT!"
print("Gold:", gold.value()) # solve returns objective value
print("successor vars:",s.value())
