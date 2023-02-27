# MAKE SURE THIS FILE IS -NOT- IN THE SQUASH COMMIT
# install brotli
# clone cpmpy-bigtest in this directory: https://github.com/CPMpy/cpmpy-bigtest 
import brotli
import glob
import os
import pickle
import time
from os.path import join
import pandas as pd

from cpmpy import *
from cpmpy.transformations.normalize import toplevel_list


def run(dirname, funcs):
    times = []
    times2 = []
    fnames = sorted(glob.glob(join(dirname, "*.bt"))+glob.glob(join(dirname, "*/*.bt")))
    print(f"{len(fnames)} models")
    for f in fnames:
        times.append(dict())
        times2.append(dict())
        with open(f, 'rb') as fpcl:
            model = pickle.loads(brotli.decompress(fpcl.read()))
            cpm_cons = model.constraints

            l = len(make_cpm_expr1b(cpm_cons))
            for func in funcs:
                t0 = time.time()
                newexp = func(cpm_cons)
                t1 = time.time() - t0
                assert (len(newexp) == l), f"Bug {func}: {len(newexp)} instead of {l}"
                times[-1][str(func.__name__)] = t1

                t2 = time.time()
                newexp2 = func(newexp)
                t3 = time.time() - t2
                assert (len(newexp2) == l), f"Bug {func}: {len(newexp)} instead of {l}"
                times2[-1][str(func.__name__)] = t3

    print("Times of first run:")
    df = pd.DataFrame.from_records(times, index=fnames)
    print("Total")
    print(df.sum().round(3).sort_values().head(8))
    print("Max")
    print(df.max().round(4).sort_values().head(4))
    print()
    print("Times of second run:")
    df = pd.DataFrame.from_records(times2, index=fnames)
    print("Total")
    print(df.sum().round(3).sort_values().head(8))
    print("Max")
    print(df.max().round(4).sort_values().head(4))

import numpy as np
from cpmpy.expressions.utils import is_any_list
from cpmpy.expressions.core import Expression, Operator, BoolVal
from cpmpy.expressions.variables import NDVarArray
from collections.abc import Iterable

# original
# 0.41 total, 0.10 max
def make_cpm_expr1(cpm_expr):
    """
        unravels nested lists and top-level AND's and ensures every element returned is a CPMpy Expression
        """

    if is_any_list(cpm_expr):
        expr = [make_cpm_expr1(e) for e in cpm_expr]
        return [e for lst in expr for e in lst]
    if cpm_expr is True:
        return []
    if cpm_expr is False:
        return [BoolVal(cpm_expr)]
    if isinstance(cpm_expr, Operator) and cpm_expr.name == "and":
        return make_cpm_expr1(cpm_expr.args)
    return [cpm_expr]

# make list upfront, append/extend to it, do inline is_any_list
# 0.27 total, 0.02 max  [not sure... something with the doulbe lists, e.g. for and]
def make_cpm_expr1b(cpm_expr):
    """
    unravels nested lists and top-level AND's and ensures every element returned is a CPMpy Expression
    """
    if cpm_expr is False:
        return [BoolVal(cpm_expr)]
    elif cpm_expr is True:
        return []
    elif isinstance(cpm_expr, Operator) and cpm_expr.name == "and":
        return [sl for e in cpm_expr.args for sl in make_cpm_expr1b(e)]
    elif isinstance(cpm_expr, (list, tuple, np.ndarray)):
        return [sl for e in cpm_expr for sl in make_cpm_expr1b(e)]
    else:
        return [cpm_expr]

# make list upfront, append/extend to it
# 0.25 total, 0.02 max
#@profile # python3 -m kernprof -l time_mkexp.py  # python3 -m line_profiler time_mkexp.py.lprof
def make_cpm_expr2(cpm_expr):
    """
    unravels nested lists and top-level AND's and ensures every element returned is a CPMpy Expression
    """
    if not is_any_list(cpm_expr):
        cpm_expr = (cpm_expr,)

    newlist = []
    for e in cpm_expr:
        if e is False:
            newlist.append(BoolVal(e))
        elif cpm_expr is True:
            pass
        elif is_any_list(e):
            newlist.extend(make_cpm_expr2(e))
        elif isinstance(e, Operator) and e.name == "and":
            newlist.extend(make_cpm_expr2(e.args))
        else:
            newlist.append(e)

    return newlist

# make list upfront, append/extend to it, do inline is_any_list
# 0.24 total, 0.02 max  [there might a tiny benefit]
#@profile # python3 -m kernprof -l time_mkexp.py  # python3 -m line_profiler time_mkexp.py.lprof
def make_cpm_expr2b(cpm_expr):
    """
    unravels nested lists and top-level AND's and ensures every element returned is a CPMpy Expression
    """
    if not isinstance(cpm_expr, (list, tuple, np.ndarray)):
        cpm_expr = (cpm_expr,)

    newlist = []
    for e in cpm_expr:
        if e is False:
            newlist.append(BoolVal(e))
        elif cpm_expr is True:
            pass
        elif isinstance(e, Operator) and e.name == "and":
            newlist.extend(make_cpm_expr2b(e.args))
        elif isinstance(e, (list, tuple, np.ndarray)):
            newlist.extend(make_cpm_expr2b(e))
        else:
            newlist.append(e)

    return newlist

# make list upfront, append/extend to it, do inline is_any_list
# 0.23 total, 0.018 max  [there might a tiny benefit, difficult to measure]
#@profile # python3 -m kernprof -l time_mkexp.py  # python3 -m line_profiler time_mkexp.py.lprof
def make_cpm_expr2bb(cpm_expr):
    """
    unravels nested lists and top-level AND's and ensures every element returned is a CPMpy Expression
    """
    if not isinstance(cpm_expr, (list, tuple, np.ndarray)):
        cpm_expr = (cpm_expr,)

    newlist = []
    for e in cpm_expr:
        if isinstance(e, (list, tuple, np.ndarray)):
            newlist.extend(make_cpm_expr2bb(e))
        elif isinstance(e, Operator) and e.name == "and":
            newlist.extend(make_cpm_expr2bb(e.args))
        elif e is False:
            newlist.append(BoolVal(e))
        elif e is not True:  # if True: pass
            newlist.append(e)

    return newlist

# some more tweaking
# 0.11 total, 0.014 max  [faster but different scale...]
def make_cpm_expr2bc(cpm_expr):
    """
    unravels nested lists and top-level AND's and ensures every element returned is a CPMpy Expression
    """
    if not isinstance(cpm_expr, (list, tuple, np.ndarray, np.flatiter)):
        cpm_expr = (cpm_expr,)

    newlist = []
    for e in cpm_expr:
        if isinstance(e, Expression):
            if isinstance(e, NDVarArray):  # sometimes does not have a .name
                newlist.extend(make_cpm_expr2bc(e.flat))
            elif e.name == "and":
                newlist.extend(make_cpm_expr2bc(e.args))
            else:
                # presumably the most frequent case
                newlist.append(e)
        elif isinstance(e, (list, tuple, np.flatiter, np.ndarray)):
            newlist.extend(make_cpm_expr2bc(e))
        elif e is False:
            newlist.append(BoolVal(e))
        elif e is not True:  # if True: pass
            newlist.append(e)

    return newlist

# some more tweaking
# 0.11 total, 0.014 max  [faster but different scale...]
def make_cpm_expr2bd(cpm_expr):
    """
    unravels nested lists and top-level AND's and ensures every element returned is a CPMpy Expression
    """
    if not isinstance(cpm_expr, Iterable):
        cpm_expr = (cpm_expr,)

    newlist = []
    for e in cpm_expr:
        if isinstance(e, Expression):
            if isinstance(e, NDVarArray):  # sometimes does not have a .name
                newlist.extend(make_cpm_expr2bd(e.flat))
            elif e.name == "and":
                newlist.extend(make_cpm_expr2bd(e.args))
            else:
                # presumably the most frequent case
                newlist.append(e)
        elif isinstance(e, Iterable):
            newlist.extend(make_cpm_expr2bd(e))
        elif e is False:
            newlist.append(BoolVal(e))
        elif e is not True:  # if True: pass
            newlist.append(e)

    return newlist

# some more tweaking
# 0.11 total, 0.014 max  [faster but different scale...]
def make_cpm_expr2be(cpm_expr):
    """
    unravels nested lists and top-level AND's and ensures every element returned is a CPMpy Expression
    """
    newlist = []

    def unravel(e):
        if isinstance(e, Expression):
            if isinstance(e, NDVarArray):  # sometimes does not have a .name
                for ee in e.flat:
                    unravel(ee)
            elif e.name == "and":
                for ee in e.args:
                    unravel(ee)
            else:
                # presumably the most frequent case
                newlist.append(e)
        elif isinstance(e, (list, tuple, np.flatiter, np.ndarray)):
            for ee in e:
                unravel(ee)
        elif e is False:
            newlist.append(BoolVal(e))
        elif e is not True:  # if True: pass
            newlist.append(e)
    unravel(cpm_expr)

    return newlist

# some more tweaking
# 0.11 total, 0.014 max  [faster but different scale...]
def make_cpm_expr2bf(cpm_expr):
    """
    unravels nested lists and top-level AND's and ensures every element returned is a CPMpy Expression
    """
    newlist = []
    def unravel(lst):
      for e in lst:
        if isinstance(e, Expression):
            if isinstance(e, NDVarArray):  # sometimes does not have a .name
                unravel(e.flat)
            elif e.name == "and":
                unravel(e.args)
            else:
                # presumably the most frequent case
                newlist.append(e)
        elif isinstance(e, (list, tuple, np.flatiter, np.ndarray)):
            unravel(e)
        elif e is False:
            newlist.append(BoolVal(e))
        elif e is not True:  # if True: pass
            newlist.append(e)
    unravel((cpm_expr,))

    return newlist


def make_cpm_expr2bg(cpm_expr):
    """
    unravels nested lists and top-level AND's and ensures every element returned is a CPMpy Expression
    """
    # check and shortcut if it will not rewrite anything, worth checking
    if isinstance(cpm_expr, (list, tuple)):
        rewrite = False
        for e in cpm_expr:
            if not isinstance(e, Expression) or \
                    isinstance(e, NDVarArray) or \
                    e.name == "and":
                rewrite = True
                break
        if not rewrite:
            return list(cpm_expr)

    # @profile
    def unravel(lst, append):
      for e in lst:
        if isinstance(e, Expression):
            if isinstance(e, NDVarArray):  # sometimes does not have a .name
                unravel(e.flat, append)
            elif e.name == "and":
                unravel(e.args, append)
            else:
                # presumably the most frequent case
                append(e)
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


def make_cpm_expr2bg2(cpm_expr):
    """
    unravels nested lists and top-level AND's and ensures every element returned is a CPMpy Expression
    """
    # check and shortcut if it will not rewrite anything, worth checking
    is_enum = isinstance(cpm_expr, (list, tuple))
    if is_enum:
        rewrite = False
        for e in cpm_expr:
            if not isinstance(e, Expression) or \
                    isinstance(e, NDVarArray) or \
                    e.name == "and":
                rewrite = True
                break
        if not rewrite:
            return list(cpm_expr)

    # @profile
    def unravel(lst, append):
      for e in lst:
        if isinstance(e, Expression):
            if isinstance(e, NDVarArray):  # sometimes does not have a .name
                unravel(e.flat, append)
            elif e.name == "and":
                unravel(e.args, append)
            else:
                # presumably the most frequent case
                append(e)
        elif isinstance(e, (list, tuple, np.flatiter, np.ndarray)):
            unravel(e, append)
        elif e is False:
            append(BoolVal(e))
        elif e is not True:  # if True: pass
            append(e)

    newlist = []
    append = newlist.append  # reuse function pointer directly
    if is_enum:
        unravel(cpm_expr, append)  # first art already enumerable
    else:
        unravel((cpm_expr,), append)  # first arg must be enumerable
    return newlist

def make_cpm_expr2bg3(cpm_expr):
    """
    unravels nested lists and top-level AND's and ensures every element returned is a CPMpy Expression
    """
    def unravel(lst, append):
      for e in lst:
        if isinstance(e, Expression):
            if isinstance(e, NDVarArray):  # sometimes does not have a .name
                unravel(e.flat, append)
            elif e.name == "and":
                unravel(e.args, append)
            else:
                # presumably the most frequent case
                append(e)
        elif isinstance(e, (list, tuple, np.flatiter, np.ndarray)):
            unravel(e, append)
        elif e is False:
            append(BoolVal(e))
        elif e is not True:  # if True: pass
            append(e)

    newlist = []
    append = newlist.append  # reuse function pointer directly

    # check and shortcut if it will not rewrite anything, worth checking
    if isinstance(cpm_expr, (list, tuple)):
        for e in cpm_expr:
            if not isinstance(e, Expression) or \
                    isinstance(e, NDVarArray) or \
                    e.name == "and":
                # needs rewrite, we know it is an enum
                unravel(cpm_expr, append)
                return newlist
        # no rewrite needed (but could be tuple)
        return list(cpm_expr)

    unravel((cpm_expr,), append)  # first arg must be enumerable
    return newlist

def make_cpm_expr2bg4(cpm_expr):
    """
    unravels nested lists and top-level AND's and ensures every element returned is a CPMpy Expression
    """
    def unravel(lst, append):
      for e in lst:
        if isinstance(e, Expression):
            if isinstance(e, NDVarArray):  # sometimes does not have a .name
                unravel(e.flat, append)
            elif e.name == "and":
                unravel(e.args, append)
            else:
                # presumably the most frequent case
                append(e)
        elif isinstance(e, (list, tuple, np.flatiter, np.ndarray)):
            unravel(e, append)
        elif e is False:
            append(BoolVal(e))
        elif e is not True:  # if True: pass
            append(e)

    newlist = []
    append = newlist.append  # reuse function pointer directly

    # check and shortcut if it will not rewrite anything, worth checking
    if isinstance(cpm_expr, (list, tuple)):
        for i,e in enumerate(cpm_expr):
            if not isinstance(e, Expression) or \
                    isinstance(e, NDVarArray) or \
                    e.name == "and":
                # keep part before as is
                unravel(cpm_expr[i:], append)
                return cpm_expr[:i]+newlist
        return list(cpm_expr)

    unravel((cpm_expr,), append)  # first arg must be enumerable
    return newlist

def make_cpm_expr6(cpm_expr):
    """
    unravels nested lists and top-level AND's and ensures every element returned is a CPMpy Expression
    """
    def unravel(lst, append):
      first = None
      for i,e in enumerate(lst):
        if isinstance(e, Expression):
            if isinstance(e, NDVarArray):  # sometimes does not have a .name
                first = i
                unravel(e.flat, append)
            elif e.name == "and":
                first = i
                unravel(e.args, append)
            else:
                # presumably the most frequent case
                if i is not None:
                    append(e)
        elif isinstance(e, (list, tuple, np.flatiter, np.ndarray)):
            first = i
            unravel(e, append)
        elif e is False:
            if i is not None:
                append(BoolVal(e))
        elif e is True:
            first = i
        elif e is not True:  # if True: pass
            if i is not None:
                append(e)
      return first

    newlist = []
    append = newlist.append  # reuse function pointer directly
    if isinstance(cpm_expr, list):
        i = unravel(cpm_expr, append)
        if i is None:
            return newlist
        elif i == len(cpm_expr):
            return cpm_expr
        return cpm_expr[:i] + newlist
    else:
        unravel((cpm_expr,), append)  # first arg must be enumerable
        return newlist

# non-recursive
def make_cpm_expr5(cpm_expr):
    """
    unravels nested lists and top-level AND's and ensures every element returned is a CPMpy Expression
    """
    l = [cpm_expr]

    i = 0
    while i < len(l):
        e = l[i]
        if isinstance(e, Expression):
            if isinstance(e, np.ndarray):  # sometimes does not have a .name
                if len(e) > 0:
                    l[i] = e[0]
                    l += e[1:].flatten()
                else:
                    # remove l[i]
                    l[i] = l[-1]
                    del l[-1]
            elif e.name == "and":
                l[i] = e.args[0]
                l += e.args[1:]
            else:
                # presumably the most frequent case
                i += 1
        elif isinstance(e, (list, tuple)):
            if len(e) > 0:
                l[i] = e[0]
                l += e[1:]
            else:
                # remove l[i]
                l[i] = l[-1]
                del l[-1]
        elif e is False:
            l[i] = BoolVal(e)
            i += 1
        elif e is True:  # if True: pass
            # remove l[i]
            l[i] = l[-1]
            del l[-1]
        else:
            i += 1

    return l
def make_cpm_expr5b(cpm_expr):
    """
    unravels nested lists and top-level AND's and ensures every element returned is a CPMpy Expression
    """
    l = [cpm_expr]

    i = 0
    while i < len(l):
        e = l[i]
        if isinstance(e, Expression):
            if isinstance(e, np.ndarray):  # sometimes does not have a .name
                l[i:i+1] = e.flatten()
            elif e.name == "and":
                l[i:i+1] = e.args
            else:
                # presumably the most frequent case
                i += 1
        elif isinstance(e, (list, tuple)):
            l[i:i+1] = e
        elif e is False:
            l[i] = BoolVal(e)
            i += 1
        elif e is True:  # if True: pass
            # remove l[i]
            del l[i]
        else:
            i += 1

    return l

# some more playing with upfront if
# 0.24 total, 0.02 max  [there might a tiny benefit]
#@profile # python3 -m kernprof -l time_mkexp.py  # python3 -m line_profiler time_mkexp.py.lprof
def make_cpm_expr2c(cpm_expr):
    """
    unravels nested lists and top-level AND's and ensures every element returned is a CPMpy Expression
    """
    if isinstance(cpm_expr, Operator) and cpm_expr.name == "and":
        cpm_expr = cpm_expr.args
    elif not isinstance(cpm_expr, (list, tuple, np.ndarray)):
        cpm_expr = (cpm_expr,)

    newlist = []
    for e in cpm_expr:
        if e is False:
            newlist.append(BoolVal(e))
        elif cpm_expr is True:
            pass
        elif isinstance(e, Operator) and e.name == "and":
            newlist.extend(make_cpm_expr2c(e.args))
        elif isinstance(e, (list, tuple, np.ndarray)):
            newlist.extend(make_cpm_expr2c(e))
        else:
            newlist.append(e)

    return newlist

# with inner loop for known enumerables (saves 1 recursive call)
# 0.29 total, 0.02 max [slightly but consistently worse on total]
def make_cpm_expr3(cpm_expr):
    """
    unravels nested lists and top-level AND's and ensures every element returned is a CPMpy Expression
    """
    if not is_any_list(cpm_expr):
        cpm_expr = (cpm_expr,)

    newlist = []
    for e in cpm_expr:
        if e is False:
            newlist.append(BoolVal(e))
        elif cpm_expr is True:
            pass
        elif is_any_list(e):
            # TODO: split up list/tuple and ndarray?
            for ee in e:
                newlist.extend(make_cpm_expr3(ee))
        elif isinstance(e, Operator) and e.name == "and":
            for ee in e.args:
                newlist.extend(make_cpm_expr3(ee))
        else:
            newlist.append(e)

    return newlist

# with inner loop for known enumerables (saves 1 recursive call)
# 0.31 total, 0.02 max [still slightly but consistently worse on total, even worse than above individual extends]
def make_cpm_expr3b(cpm_expr):
    """
    unravels nested lists and top-level AND's and ensures every element returned is a CPMpy Expression
    """
    if not is_any_list(cpm_expr):
        cpm_expr = (cpm_expr,)

    newlist = []
    for e in cpm_expr:
        if e is False:
            newlist.append(BoolVal(e))
        elif cpm_expr is True:
            pass
        elif is_any_list(e):
            # TODO: split up list/tuple and ndarray?
            newlist.extend(make_cpm_expr3(ee) for ee in e)
        elif isinstance(e, Operator) and e.name == "and":
            for ee in e.args:
                newlist.extend(make_cpm_expr3(ee) for ee in e.args)
        else:
            newlist.append(e)

    return newlist

# make list upfront, append/extend to it, copy upfront
# 0.33 total, 0.02 max [worse... probably the inplace moves]
def make_cpm_expr4(cpm_expr):
    """
    unravels nested lists and top-level AND's and ensures every element returned is a CPMpy Expression
    """
    if not is_any_list(cpm_expr):
        cpm_expr = (cpm_expr,)

    newlist = list(cpm_expr) # take copy
    for i,e in enumerate(newlist):
        if e is False:
            newlist[i] = BoolVal(e)
        elif cpm_expr is True:
            del newlist[i]
        elif is_any_list(e):
            newlist[i:i+1] = make_cpm_expr4(e)
        elif isinstance(e, Operator) and e.name == "and":
            newlist[i:i+1] = make_cpm_expr4(e.args)

    return newlist

####### Ignace's attempts ########
def make_cpm_expr_6(lst):
    # clean non-iterative version,
    # slower because of inline modification of lists
    if not is_any_list(lst):
        lst = [lst]
    i = 0
    lst = list(lst)
    while i < len(lst):
        expr = lst[i]
        if is_any_list(expr):
            lst.extend(lst.pop(i))
        elif isinstance(expr, Operator) and lst[i].name == "and":
            lst.extend(lst.pop(i).args)
        elif lst is False:
            lst.append(BoolVal(lst.pop(i)))
        else:
            # fine, skip
            i += 1

    return lst

def make_cpm_expr_7(lst):
    # non-recursive version without inline changing size of list
    # faster than previous but does not beat 'make_cpm_expr_2bf'
    # main time lost in if-checks (and a little bit in the "out.append" where it would be better to use out.extend with a bunch of elements in 1 go)

    if not is_any_list(lst):
        lst = [lst]
    out, stack = [], [lst]
    while len(stack):
        expr = stack.pop()
        if isinstance(expr, Expression):
            if isinstance(expr, np.ndarray):  # sometimes does not have a .name
                stack.extend(expr.flat)
            elif expr.name == "and":
                stack.extend(expr.args)
            else:
                out.append(expr)
        if is_any_list(expr):
            stack.extend(expr)
        elif isinstance(expr, Operator) and expr.name == "and":
            stack.extend(expr.args)
        elif expr is False:
            out.append(BoolVal(expr))
        elif expr is not True:  # if True: pass
            out.append(expr)

    out.extend(stack)
    return out

def make_cpm_expr_7b(lst):
    if not is_any_list(lst):
        lst = [lst]
    out, stack = [], [lst]
    i = 1
    while i <= len(stack):
        if is_any_list(stack[len(stack)-i]):
            # last to len(stack)-i+1 are normal exprs
            out.extend(stack[len(stack)-i+1:])
            stack[len(stack)-i+1:] = []
            i = 1

            stack.extend(stack.pop())
        elif isinstance(stack[len(stack)-i], Operator) and stack[len(stack)-i].name == "and":
            # last to len(stack)-i+1 are normal exprs
            out.extend(stack[len(stack) - i + 1:])
            stack[len(stack) - i + 1:] = []
            i = 1

            stack.extend(stack.pop().args)

        elif stack[len(stack)-i] is False:
            # last to len(stack)-i+1 are normal exprs
            out.extend(stack[len(stack) - i + 1:])
            stack[len(stack) - i + 1:] = []
            i = 1

            out.append(BoolVal(stack.pop()))
        else:
            i += 1
    out.extend(stack)
    return out



# @profile
def make_cpm_expr_generator(cpm_expr):
    """
    unravels nested lists and top-level AND's and ensures every element returned is a CPMpy Expression
    """
    if not isinstance(cpm_expr, (list, tuple, np.ndarray, np.flatiter)):
        cpm_expr = (cpm_expr,)

    # @profile
    def do_recurse(cpm_expr):

        for e in cpm_expr:
            if isinstance(e, Expression):
                if isinstance(e, NDVarArray):  # sometimes does not have a .name
                    yield from do_recurse(e.flat)
                elif e.name == "and":
                    yield from do_recurse(e.args)
                else:
                    # presumably the most frequent case
                    yield e
            elif isinstance(e, (list, tuple, np.flatiter, np.ndarray)):
                yield from do_recurse(e)
            elif e is False:
                yield BoolVal(e)
            elif e is not True:  # if True: pass
                yield e

    return list(do_recurse(cpm_expr))


# @profile
def make_cpm_expr2bf_ignace(cpm_expr):
    """
    unravels nested lists and top-level AND's and ensures every element returned is a CPMpy Expression
    """
    newlist = []
    append = newlist.append
    # @profile
    def unravel(lst):
      for e in lst:
        if isinstance(e, Expression):
            if isinstance(e, NDVarArray):  # sometimes does not have a .name
                unravel(e.flat)
            elif e.name == "and":
                unravel(e.args)
            else:
                # presumably the most frequent case
                append(e)
        elif isinstance(e, (list, tuple, np.flatiter, np.ndarray)):
            unravel(e)
        elif e is False:
            append(BoolVal(e))
        elif e is not True:  # if True: pass
            append(e)
    unravel((cpm_expr,))

    return newlist

# add func as arg? (local lookup)
def make_cpm_expr2bf_ignace2(cpm_expr):
    """
    unravels nested lists and top-level AND's and ensures every element returned is a CPMpy Expression
    """
    # @profile
    def unravel(lst, append):
      for e in lst:
        if isinstance(e, Expression):
            if isinstance(e, NDVarArray):  # sometimes does not have a .name
                unravel(e.flat, append)
            elif e.name == "and":
                unravel(e.args, append)
            else:
                # presumably the most frequent case
                append(e)
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


if __name__ == '__main__':
    dirname = os.path.join("cpmpy-bigtest","models")
    assert os.path.exists("cpmpy-bigtest"), "Make sure you cloned bigtest in `cpmpy-bigtest/`"

    funcs = [
        make_cpm_expr1,
        # make_cpm_expr1b,
        # make_cpm_expr2,
        make_cpm_expr2b,
        #make_cpm_expr2bb,
        make_cpm_expr2bc,
        #make_cpm_expr2bd,
        #make_cpm_expr2be,
        #make_cpm_expr2bf,
        #make_cpm_expr2bg,
        #make_cpm_expr2bg2,
        #make_cpm_expr2bg3,
        #make_cpm_expr2bg4,
        make_cpm_expr2c,
        # make_cpm_expr3,
        # make_cpm_expr3b,
        # make_cpm_expr4,
        # make_cpm_expr5,
        # make_cpm_expr5b,
        #make_cpm_expr_6,
        # make_cpm_expr_7,  # buggy
        make_cpm_expr_7b,
        make_cpm_expr_generator,
        #make_cpm_expr2bf_ignace,
        make_cpm_expr2bf_ignace2,
        toplevel_list,

    ]

    run(dirname, funcs)
