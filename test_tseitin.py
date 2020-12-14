#!/usr/bin/python3
"""
The 'frietkot' problem; invented by Tias to explain SAT and SAT solving
http://homepages.vub.ac.be/~tiasguns/frietkot/
"""
from cppy import *
from cppy.model_tools.to_cnf import *
import pandas as pd

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
print("a->((b&c)|(d&e))", to_cnf([ implies(a, ((b&c)|(d&e))) ]))

# more advanced tseitin tests
class Relation(object):
    # rows, cols: list of names
    def __init__(self, rows, cols):
        rel = BoolVar((len(rows), len(cols)))
        self.df = pd.DataFrame(index=rows, columns=cols)
        for i,r in enumerate(rows):
            for j,c in enumerate(cols):
                self.df.loc[r,c] = rel[i,j]
    # use as: rel['a','b']
    def __getitem__(self, key):
        try:
            return self.df.loc[key]
        except KeyError:
            return False

material = ['mat1']
spot = ['1']
game = ['cricket']
mat_spot = Relation(material, spot)
spot_game = Relation(spot, game)
mat_game = Relation(material, game)

for m in material:
    for s in spot:
        for g in game:
            print("a->((b&c)->d)", to_cnf( implies( a, implies( mat_spot[m, s] & spot_game[s, g], mat_game[m, g] )) )    )


# implies implies tests
print("a->((b&~c)->~d)", to_cnf( implies( a, implies( ~b & c, ~d) ))  )
print("a->((~b&c)->~d)", to_cnf( implies( a, implies( b & ~c, ~d) )) )

# double impliciation tests (with '==')
print("c | (a -> b & b -> a)", to_cnf(c | (implies(a,b) & implies(b,a))))
print("c | (a <-> b)", to_cnf(c | (a == b)))
print("a <-> ~b", to_cnf(a == ~b))
print("a <-> True", to_cnf(a == True))
print("False <-> a", to_cnf(False == a))
print("a <-> ~b|c", to_cnf(a == (~b|c)))
print("b|c|-d <-> a", to_cnf((b|c|-d) == a))
