import numpy as np

from ..expressions.core import BoolVal, Expression
from ..expressions.variables import NDVarArray

def make_cpm_expr(cpm_expr):
    """
    unravels nested lists and top-level AND's and ensures every element returned is a CPMpy Expression
    """
    # very efficient version with limited function lookups and list operations
    def unravel(lst, append):
      for e in lst:
        if isinstance(e, Expression):
            if isinstance(e, NDVarArray):  # sometimes does not have a .name
                unravel(e.flat, append)
            elif e.name == "and":
                unravel(e.args, append)
            else:
                append(e) # presumably the most frequent case
        elif isinstance(e, (list, tuple, np.flatiter, np.ndarray)):
            unravel(e, append)
        elif e is False:
            append(BoolVal(e))
        elif e is not True:  # if True: pass
            append(e)

    newlist = []
    append = newlist.append
    unravel((cpm_expr,), append)

    return newlist
