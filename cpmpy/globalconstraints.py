from .expressions import *
from itertools import chain, combinations
# in one file for easy overview, does not include interpretation
def all_pairs(args):
    pairs = list(combinations(args, 2))
    return pairs

class allequal(GlobalConstraint):
    def __init__(self,  arg_list,name="allequal"):
        super().__init__(name, arg_list)
    
    def decompose(self):
        return [var1 == var2 for var1, var2 in all_pairs(self.args)]
class alldifferent(GlobalConstraint):
    def __init__(self, arg_list, name="alldifferent"):
        super().__init__(name, arg_list)
    
    def decompose(self):
        return [var1 != var2 for var1, var2 in all_pairs(self.args)]
class circuit(GlobalConstraint):
    def __init__(self, arg_list, name="circuit"):
        super().__init__(name, arg_list)
    
    def decompose(self):
        n = len(self.args)
        z = IntVar(0, n-1, n)
        constraints = []
        constraints +=alldifferent.decompose(z)
        constraints +=alldifferent.decompose(self.args)
        constraints += [z[0]==self.args[0]]
        constraints += [z[n-1]==self.args[0]]
        for i in range(1,n-1):
            constraints+= [z[i] == self.args[z[i-1]]]
            constraints+= [z[i] != 0]


        return constraints



