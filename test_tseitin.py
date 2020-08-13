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
print("a|b:", tseitin([a | b]) )
print("a&b:", tseitin([a & b]) )
print("a->b:", tseitin([implies(a,b)]) )
print("a|b|c", tseitin([a | b | c]) )
print("a&b&c&d:", tseitin([a & b & c & d]) )
print("a&(c|d):", tseitin([ a&(c|d) ]))
print("a|b, a&(c|d):", tseitin([ a|b, a&(c|d) ]))
print("a&b, a|(c&d):", tseitin([ a&b, a|(c&d) ]))
print("((a|b)&c -> ~d): (wikipedia)", tseitin([ implies((a|b)&c, ~d) ]))
print("a->~b, b|~c|d, c->((a&b)|d), a&(b|(c&(d|e)))", tseitin([ implies(a, ~b), b|~c|d, implies(c, ((a&b)|d)), a&(b|(c&(d|e))) ]))
