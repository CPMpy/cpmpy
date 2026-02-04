"""
Transform constraints to **Conjunctive Normal Form** (i.e. an `and` of `or`s of literals, i.e. Boolean variables or their negation, e.g. from `x xor y` to `(x or ~y) and (~x or y)`) using a back-end encoding library and its transformation pipeline.
"""

import cpmpy as cp
from ..expressions.variables import _BoolVarImpl
from ..expressions.core import Operator
from ..solvers.pindakaas import CPM_pindakaas
from cpmpy.tools.explain.marco import make_assump_model
from cpmpy.expressions.utils import all_pairs


def to_cnf(constraints, csemap=None, ivarmap=None, encoding="auto"):
    """
    Converts all constraints into **Conjunctive Normal Form**

    Arguments:
        constraints:    list[Expression] or Operator
        csemap:         `dict()` used for CSE
        ivarmap:        `dict()` used to map integer variables to their encoding (usefull for finding the values of the now-encoded integer variables)
        encoding:       the encoding used for `int2bool`, choose from ("auto", "direct", "order", or "binary")
    Returns:
        Equivalent CPMpy constraints in CNF, and the updated `ivarmap`
    """
    if not CPM_pindakaas.supported():
        raise ImportError(
            f"Install the Pindakaas python library `pindakaas` (e.g. `pip install pindakaas`) package to use the `to_cnf` transformation"
        )

    import pindakaas as pdk

    slv = CPM_pindakaas()
    slv.encoding = encoding

    if ivarmap is not None:
        slv.ivarmap = ivarmap
    slv._csemap = csemap

    # the encoded constraints (i.e. `PB`s) will be added to this `pdk.CNF` object
    slv.pdk_solver = pdk.CNF()

    # add, transform, and encode constraints into CNF/clauses
    slv += constraints

    # now we read the pdk.CNF back to cpmpy constraints by mapping from `pdk.Lit` to CPMpy lit
    cpmpy_vars = {str(slv.solver_var(x).var()): x for x in slv._int2bool_user_vars()}

    # if a user variable `x` does not occur in any clause, it should be added as `x | ~x`
    free_vars = set(cpmpy_vars.values())

    def to_cpmpy_clause(clause):
        """Lazily convert `pdk.CNF` to CPMpy."""
        for lit in clause:
            x = str(lit.var())
            if x not in cpmpy_vars:
                cpmpy_vars[x] = cp.boolvar()
            elif cpmpy_vars[x] in free_vars:  # cpmpy_vars[x] is only in free_vars if it existed before
                free_vars.remove(cpmpy_vars[x])
            yield ~cpmpy_vars[x] if lit.is_negated() else cpmpy_vars[x]

    clauses = []
    clauses += (cp.any(to_cpmpy_clause(clause)) for clause in slv.pdk_solver.clauses())
    clauses += ((x | ~x) for x in free_vars)  # add free variables so they are "known" by the CNF

    return clauses


def to_gcnf(soft, hard=None, name=None, csemap=None, ivarmap=None, encoding="auto", normalize=False):
    """
    Or `make_assump_cnf`; returns an assumption CNF model, and separately the soft clauses, hard clauses, and assumption variables. Follows https://satisfiability.org/competition/2011/rules.pdf, however, there is no guarantee that the groups are disjoint.
    """

    model, soft_, assump = make_assump_model(soft, hard=hard, name=name)
    cnf = to_cnf(model.constraints, encoding=encoding, csemap=csemap, ivarmap=ivarmap)

    constraints = {
        True: [],  # hard clauses
        **{a: [] for a in assump},  # assumption mapped to its soft clauses
    }

    def add_gcnf_clause_(lits):
        # find the assumption variable (not guaranteed to be first)
        i = next((i for i, l in enumerate(lits) if (~l) in assump), None)
        if i:
            constraints[~lits[i]].append(cp.any(l for i_, l in enumerate(lits) if i_ != i))
        else:
            # hard clause (w/o assumption var)
            constraints[True].append(cp.any(lits))

    def add_gcnf_clause(cpm_expr):
        for clause in _to_clauses(cpm_expr):
            i = next((i for i, l in enumerate(clause) if (~l) in assump), None)
            if i is None:
                # hard clause (w/o assumption var)
                constraints[True].append(cp.any(clause))
            else:
                # soft clause
                constraints[~clause[i]].append(cp.any(l for i_, l in enumerate(clause) if i_ != i))

    for c in cnf:
        add_gcnf_clause(c)

    if normalize:
        # to make groups disjoint..
        for (a, g_a), (b, g_b) in all_pairs(constraints.items()):
            for i, c_a in enumerate(g_a):
                for j, c_b in enumerate(g_b):
                    # TODO efficiency, plus account for shuffled literals
                    # ..we find shared clauses between any two groups..
                    if c_a == c_b:
                        # ..in the second group, we replace the clause `c_b` for unit clause `f`
                        f = cp.boolvar()
                        g_b[j] = f
                        # then add `f -> c_b` as a hard clause
                        add_gcnf_clause(f.implies(c_b))

        for g_a, g_b in all_pairs(constraints.values()):
            for c_a in g_a:
                for c_b in g_b:
                    assert not (c_a == c_b), "Still not disjoint"



    return (
        cp.Model(cnf),
        [cp.all(constraints[a]) for a in assump],
        [cp.all(constraints[True])] if constraints[True] else [],
        assump,
    )


def _to_clauses(cons):
    """Takes some CPMpy constraints in CNF + half-reifications and returns clauses as list of lists"""
    if isinstance(cons, _BoolVarImpl):
        return [[cons]]
    elif isinstance(cons, Operator):
        if cons.name == "or":
            return [cons.args]
        elif cons.name == "and":
            return [c_ for c in cons.args for c_ in _to_clauses(c)]
        elif cons.name == "->":
            return [[~cons.args[0], *c] for c in _to_clauses(cons.args[1])]
        else:
            raise NotImplementedError(f"Unsupported Op {cons.name}")
    elif cons is True:
        return []
    else:
        raise NotImplementedError(f"Unsupported constraint {cons}")
