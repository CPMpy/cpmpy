#!/usr/bin/python3
"""
The 'frietkot' problem; invented by Tias to explain SAT and SAT solving
http://homepages.vub.ac.be/~tiasguns/frietkot/
"""
from cppy import *
from cppy.model_tools.to_cnf import *

# Construct the model.
(a,b,c,d,e) = BoolVar(5)

print("empty:", to_cnf([]) )
print("a|b:", to_cnf([a | b]) )
print("a&b:", to_cnf([a & b]) )
print("a->b:", to_cnf([implies(a,b)]) )
print("a|b|c", to_cnf([a | b | c]) )
print("a&b&c&d:", to_cnf([a & b & c & d]) )
print("a&(c|d):", to_cnf([ a&(c|d) ]))
print("a|b, a&(c|d):", to_cnf([ a|b, a&(c|d) ]))
print("a&b, a|(c&d):", to_cnf([ a&b, a|(c&d) ]))
print("((a|b)&c -> ~d): (wikipedia)", to_cnf([ implies((a|b)&c, ~d) ]))
print("a->~b, b|~c|d, c->((a&b)|d), a&(b|(c&(d|e)))", to_cnf([ implies(a, ~b), b|~c|d, implies(c, ((a&b)|d)), a&(b|(c&(d|e))) ]))
