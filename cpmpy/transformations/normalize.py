import numpy as np

from ..expressions.core import BoolVal, Expression
from ..expressions.variables import NDVarArray
from ..exceptions import NotSupportedError

def toplevel_list(cpm_expr, merge_and=True):
    """
    unravels nested lists and top-level AND's and ensures every element returned is a CPMpy Expression with `.is_bool()` true.

    - cpm_expr: Expression or list of Expressions
    - merge_and: if True then a toplevel 'and' will have its arguments merged at top level
    """
    # very efficient version with limited function lookups and list operations
    def unravel(lst, append):
        for e in lst:
            if isinstance(e, Expression):
                if isinstance(e, NDVarArray):  # sometimes does not have .name
                    unravel(e.flat, append)
                elif merge_and and e.name == "and":
                    unravel(e.args, append)
                else:
                    assert (e.is_bool()), f"Only boolean expressions allowed at toplevel, got {e}"
                    append(e) # presumably the most frequent case
            elif isinstance(e, (list, tuple, np.flatiter, np.ndarray)):
                unravel(e, append)
            elif e is False:
                append(BoolVal(e))
            elif e is not True:  # if True: pass
                raise NotSupportedError(f"Expression {e} is not a valid CPMpy constraint")

    newlist = []
    append = newlist.append
    unravel((cpm_expr,), append)

    return newlist
