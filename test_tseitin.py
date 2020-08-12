#!/usr/bin/python3
"""
The 'frietkot' problem; invented by Tias to explain SAT and SAT solving
http://homepages.vub.ac.be/~tiasguns/frietkot/
"""
from cppy import *
from cppy.model_tools.transforms import *

# Construct the model.
(a,b,c,d,e) = BoolVar(5)

print("empty:", tseitin([]) )
print("or:", tseitin([a | b]) )
print("and:", tseitin([a & b]) )
print("impl:", tseitin([implies(a,b)]) )
print("or3a:", tseitin(a | b | c) )
print("or3:", tseitin([a | b | c]) )
print("and4:", tseitin([a & b & c & d]) )
print("a&(c|d):", tseitin([ a&(c|d) ]))
print("a|b, a&(c|d):", tseitin([ a|b, a&(c|d) ]))
print("a&b, a|(c&d):", tseitin([ a&b, a|(c&d) ]))
print("((a|b)&c -> ~d): (wikipedia)", tseitin([ implies((a|b)&c, ~d) ]))
