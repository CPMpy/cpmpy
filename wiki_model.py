#!/usr/bin/python3
"""
Wikipedia's example when explaining Tseitin transformation:
https://en.wikipedia.org/wiki/Tseytin_transformation
"""
# import sys
from cppy import *

(p,q,r,s) = BoolVar(4)

formula = implies( ((p | q) & r), ~s )
print("Formula:")
print(formula)

cnf, new_vars = Model([ formula ]).to_cnf()
print("\nCNF:")
print(cnf)
print("vars")
print(new_vars)


