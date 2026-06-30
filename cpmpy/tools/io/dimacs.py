#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## dimacs.py
##
"""
Helper functions for loading CPMpy models from and writing to DIMACS format.

DIMACS is a textual format to represent CNF problems.
More can be read about it here: 
- https://satisfiability.org/competition/2009/format-benchmarks2009.html
- https://people.sc.fsu.edu/~jburkardt/data/cnf/cnf.html


Format:
=======

The header of the file can optionally include a p-line; ``p cnf <n_vars> <n_constraints>``.
If the number of variables and constraints are not given, it is inferred by the parser.

.. note::

    It is not preferred by the SAT competition to no longer include the p-line.

Each remaining line of the file is formatted as a list of integers with a trailing 0; literals belonging to the same clause.
An integer represents a Boolean variable and a negative Boolean variable is represented using a `'-'` sign.

E.g. the clause ``(a or b or c)`` is represented as ``1 2 3 0``.

Comments are lines starting with a `c` character.

Full example:

.. code-block:: text

    c This is a comment
    p cnf 3 3
    1 2 3 0
    c This is another comment
    -2 -3 0
    -1 0
    
=================
List of functions
=================

.. autosummary::
    :nosignatures:

    load_dimacs
    write_dimacs
"""

import os
import builtins
import warnings
from typing import TextIO
from typing import Optional, Callable, Union

import cpmpy as cp

from cpmpy.expressions.variables import _BoolVarImpl, NegBoolView, NDVarArray
from cpmpy.expressions.core import Operator

from cpmpy.transformations.to_cnf import to_cnf, to_cnf_objective
from cpmpy.transformations.get_variables import get_variables
from cpmpy.transformations.cse import CSEMap
from cpmpy.transformations.int2bool import IntVarEnc
from cpmpy.tools.io.utils import _create_header, _handle_loader_input


def write_dimacs(
        model: cp.Model, 
        path: Optional[Union[str, os.PathLike]] = None, 
        encoding: str = "auto", 
        p_header: bool = False, header : Optional[str] = None, 
        open: Callable = builtins.open, 
        annotate: Optional[Callable] = None
    ) -> str:
    """
    Writes a CPMpy model to DIMACS format.
    Uses the "to_cnf" transformation from CPMpy.

    .. note::
        If the model has an objective, WCNF is emitted. DIMACS/WCNF has no field for
        constant objective offsets; when objective transformation introduces one, it
        is ignored and a warning is raised. The written model still preserves the
        optimisation, but its objective value may differ by that constant.

    Arguments:
        model (cp.Model): a CPMpy model
        path (str or os.PathLike, optional): file path to write the DIMACS output to. If None, the DIMACS string is returned.
        encoding (str): the encoding used for `int2bool`, choose from ("auto", "direct", "order", or "binary") (default: "auto")
        p_header (bool): whether to include the ``p ...`` problem header line. Replaces the ``h`` prefix for WCNF with a ``top`` weight. (default: ``False``)
        header (str, optional): Optional header text to prepend as DIMACS comments.
            If None, a default CPMpy header is created only when writing to ``path``.
            Pass an empty string to skip adding a header.
        open (Callable): callable to open the file for writing (default: builtin ``open``).
            Called as ``open(path, "w")``. This mirrors the ``open=`` argument
            in loaders and allows custom compression or I/O (e.g.
            ``lambda p, mode='w': lzma.open(p, 'wt')``).
        annotate (Callable, optional): variable annotation strategy. Controls how DIMACS literal IDs are
            mapped back to original CPMpy variables. 
            Options:
            - None (default): no annotation
            - "dimacs_comments": Sugar-style 'c <id> <name>' comment lines (self-contained)
            - "json_sidecar": comments + a .map.json sidecar file (BumbleBee pattern)
            - VariableAnnotator instance: fully custom strategy
    """

    if header is None:
        header = _create_header(format="wcnf" if model.has_objective() else "cnf") if path is not None else None
    elif header == "":
        header = None

    # Shared maps so both objective and constraint transformations populate
    # the same ivarmap, enabling annotation of all integer variable encodings.
    ivarmap: dict[str, IntVarEnc] = dict()
    csemap = CSEMap()

    constraints = list.copy(model.constraints)
    objective_lits = []
    objective_weights = []
    objective_const = 0

    # Transform objective, if present
    if model.has_objective():
        objective_weights, objective_lits, objective_const, extra_cons = to_cnf_objective(
            model.objective_, encoding=encoding, csemap=csemap, ivarmap=ivarmap
        )
        if objective_const != 0:
            warnings.warn(
                "DIMACS/WCNF cannot represent constant objective offsets; "
                f"ignoring offset {objective_const}.",
                UserWarning,
                stacklevel=2,
            )
        # Add constraints resulting from the objective transformation
        constraints += extra_cons
    # Transform constraints to CNF
    constraints = to_cnf(constraints, csemap=csemap, ivarmap=ivarmap, encoding=encoding)

    # Variable to DIMACS literal ID mapping
    vars = get_variables(constraints + objective_lits)
    mapping = {v : i+1 for i, v in enumerate(vars)}

    top_weight = sum(objective_weights) + 1 if objective_weights else 1
    if model.has_objective():
        hard_prefix = f"{top_weight} " if p_header else "h "
    else:
        hard_prefix = ""

    out = ""  # DIMACS string

    # Write constraints to DIMACS format
    for cons in constraints:

        # Collect literals in clause
        literals: list[NegBoolView | _BoolVarImpl]
        if isinstance(cons, _BoolVarImpl):
            literals = [cons]
        else:
            if not (isinstance(cons, Operator) and cons.name == "or"):
                raise NotImplementedError(f"Unsupported constraint {cons}")
            literals = list(cons.args)


        # Write clause to DIMACS format
        dimacs_clause_ints = []
        for l in literals:
            if isinstance(l, NegBoolView):
                dimacs_clause_ints.append(str(-mapping[l._bv]))
            elif isinstance(l, _BoolVarImpl):
                dimacs_clause_ints.append(str(mapping[l]))
            else:
                raise ValueError(f"Expected Boolean variable in clause, but got {l} which is of type {type(l)}")

        out += hard_prefix + " ".join(dimacs_clause_ints + ["0"]) + "\n"

    def _dimacs_lit(lit, flip_sign=False):
        sign = -1 if flip_sign else 1
        if isinstance(lit, NegBoolView):
            return str(-sign * mapping[lit._bv])
        if isinstance(lit, _BoolVarImpl):
            return str(sign * mapping[lit])
        raise ValueError(f"Expected Boolean literal in objective, but got {lit} of type {type(lit)}")

    # Write objective to DIMACS format
    if model.has_objective():
        for w, x in zip(objective_weights, objective_lits):
            # WCNF minimizes the weight of unsatisfied soft clauses. Each objective
            # literal is written as a one-literal soft clause: for minimization we
            # flip the literal sign, so the penalty is paid when the original
            # literal is true; for maximization we keep the sign.
            lit = _dimacs_lit(x, flip_sign=model.objective_is_min)
            out += f"{w} {lit} 0\n"

    # Write annotations to DIMACS string
    if annotate is not None:
        if callable(annotate):
            comments = annotate(vars, ivarmap)
            if comments:
                comment_block = "\n".join(f"c {i+1} " + c for i,c in enumerate(comments)) + "\n"
                out = comment_block + out
        else:
            raise ValueError(f"Expected a Callable annotate, but got {type(annotate)}")

    # Optional p-header
    if p_header:
        if model.has_objective():
            nr_clauses = len(constraints) + len(objective_weights)
            out = f"p wcnf {len(vars)} {nr_clauses} {top_weight}\n" + out
        else:
            out = f"p cnf {len(vars)} {len(constraints)}\n" + out

    # Optional header
    if header is not None:
        header_lines = ["c " + line for line in header.splitlines()]
        out = "\n".join(header_lines) + "\n" + out

    # Write to file
    if path is not None:
        with open(path, "w") as f:
            f.write(out)

    return out


def load_dimacs(dimacs: Union[str, os.PathLike, TextIO], open: Callable = builtins.open, type: Optional[str] = None):
    """
    Load a CPMpy model from a DIMACS formatted file strictly following the specification.

    .. note::

        The (optional) p-line has to denote the correct number of variables and clauses.

    Arguments:
        dimacs (str or os.PathLike or TextIO):
            - A file path to a DIMACS/WCNF file, or
            - A string containing DIMACS/WCNF content directly, or
            - A TextIO object already open for reading
        open (Callable): callable to open the file for reading (default: builtin ``open``).
            Use for decompression, e.g. ``lambda p: lzma.open(p, 'rt')`` for ``.cnf.xz``.
        type (str, optional): type of the file to load. If None, it is inferred from the file content.
            Supported types: "cnf", "wcnf".

    Returns:
        cp.Model: The CPMpy model of the DIMACS instance.

    Raises:
        ValueError: If the optional type argument is not supported.
    """

    with _handle_loader_input(dimacs, open=open) as f:

        # No type hint provided -> auto-detect type
        if type is None:

            # Auto-detect weighted instances:
            # - explicit `p wcnf ...` header
            # - any hard-clause line starting with `h`
            # - no header but all non-comment clause lines look weighted (weight literals... 0)
            is_weighted = False
            weighted_compatible = True
            saw_clause_line = False
            for raw in f:
                line = raw.strip()
                if line == "" or line.startswith("c"):
                    continue
                if line.startswith("p"):
                    params = line.split()
                    assert len(params) >= 4, f"Expected p-header to be formed `p <typ> ...` but got {line}"
                    _, typ, *_ = params
                    if typ == "wcnf":
                        is_weighted = True
                    elif typ != "cnf":
                        raise ValueError(f"Expected `cnf` or `wcnf` as file format, but got {typ} which is not supported.")
                    break
                if line.startswith("h"):
                    is_weighted = True
                    break
                saw_clause_line = True
                try:
                    ints = [int(tok) for tok in line.split()]
                except ValueError:
                    weighted_compatible = False
                    continue
                if len(ints) < 2 or ints[-1] != 0 or ints[0] < 0:
                    weighted_compatible = False

            if not is_weighted and saw_clause_line and weighted_compatible:
                is_weighted = True
            
        # Type hint provided -> use it
        elif type == "wcnf":
            is_weighted = True
        elif type == "cnf":
            is_weighted = False
        else:
            raise ValueError(f"Expected `cnf` or `wcnf` as optional type argument, but got {type} instead.")

        f.seek(0)

        # If weighted, delegate to WCNF loader
        if is_weighted:
            from cpmpy.tools.io.wcnf import load_wcnf
            return load_wcnf(f)

        # -------------------------------- CNF parser -------------------------------- #

        # CNF parse (strict with p-line counts when present, inferred otherwise)
        m = cp.Model()
        clause: list[int] = []
        clauses = []
        nr_vars_declared = None
        nr_cls_declared = None
        max_var = 0

        for raw in f:
            line = raw.strip()
            if line == "" or line.startswith("c"):
                continue  # skip empty and comment lines
            if line.startswith("p"):
                params = line.split()
                assert len(params) == 4, f"Expected p-header to be formed `p cnf nr_vars nr_cls` but got {line}"
                _, typ, nr_vars_text, nr_cls_text = params
                if typ != "cnf":
                    raise ValueError(f"Expected `cnf` (i.e. DIMACS) as file format, but got {typ} which is not supported.")
                nr_vars_declared = int(nr_vars_text)
                nr_cls_declared = int(nr_cls_text)
                continue

            for token in line.split():
                i = int(token)
                if i == 0: # end of clause
                    clauses.append(clause)
                    clause = []
                else:
                    max_var = max(max_var, abs(i)) # keep running max literal ID
                    clause.append(i)

        assert len(clause) == 0, "Expected last clause to be terminated by 0"

        nr_vars = nr_vars_declared if nr_vars_declared is not None else max_var
        if nr_vars_declared is not None:
            assert max_var <= nr_vars_declared, f"Expected at most {nr_vars_declared} variables (from p-line) but found literal index {max_var}"

        bvs: NDVarArray = cp.boolvar(shape=(nr_vars,))
        for cl in clauses:
            if len(cl) == 0:
                m += cp.BoolVal(False)
                continue
            lits = []
            for lit_id in cl:
                bv = bvs[abs(lit_id)-1]
                lits.append(bv if lit_id > 0 else ~bv)
            m += cp.any(lits)

        if nr_cls_declared is not None:
            assert len(m.constraints) == nr_cls_declared, f"Number of clauses was declared in p-line as {nr_cls_declared}, but was {len(m.constraints)}"

        return m
