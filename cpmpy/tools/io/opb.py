#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## opb.py
##
"""
OPB parser.

Currently only the restricted OPB PB24 format is supported (without WBO).
More can be read about it here:

- https://www.cril.univ-artois.fr/PB24/OPBgeneral.pdf


=================
List of functions
=================

.. autosummary::
    :nosignatures:

    load_opb
    write_opb

"""


import os
import re
import builtins
from typing import Union, Optional, Callable, Any, TextIO
from functools import reduce
from operator import mul
from functools import partial

import cpmpy as cp
from cpmpy.transformations.cse import CSEMap
from cpmpy.transformations.get_variables import get_variables
from cpmpy.transformations.to_opb import to_opb, to_opb_objective
from cpmpy.expressions.variables import NegBoolView, _ignore_strict_variable_name_check
from cpmpy.expressions.core import Operator, Comparison
from cpmpy.model import _update_variable_counters
from cpmpy.tools.io.annotate_bool import BooleanEncodingAnnotator
from cpmpy.tools.io.utils import _create_header, _handle_loader_input


# Regular expressions
HEADER_RE = re.compile(r'(.*)\s*#variable=\s*(\d+)\s*#constraint=\s*(\d+).*')
TERM_RE = re.compile(r"([+-])\s*((?:(?:\d+\s+)?~?[^\s;]+(?:\s+~?[^\s;]+)*))(?=\s+[+-]|$)")
TERM_RE = re.compile(r"([+-]?)\s*((?:(?:\d+\s+)?~?[^\s;=]+(?:\s+~?[^\s;=]+)*?))(?=\s*[+-]|\s*=|\s*;|$)")
OBJ_TERM_RE = re.compile(r'^min:')
IND_TERM_RE = re.compile(r'([>=|<=|=]+)\s+([+-]?\d+)')
IND_TERM_RE = re.compile(r'(>=|<=|=)\s*([+-]?\d+)')


def _parse_term(line, vars):
    """
    Parse a line containing OPB terms into a CPMpy expression.

    Supports:
        - Linear terms (e.g., +2 x1)
        - Non-linear terms (e.g., -1 x1 x14)
        - Negated variables using '~' (e.g., ~x5)

    Arguments:
        line (str):                 A string containing one or more terms.
        vars (list[cp._BoolVarImpl]):    List or array of CPMpy Boolean variables.

    Returns:
        cp.Expression: A CPMpy expression representing the sum of all parsed terms.

    Example:
        >>> vars = list(cp.boolvars(5))
        >>> _parse_term("2 x2 x3 +3 x4 ~x5", vars)
        sum([2, 3] * [(IV2*IV3), (IV4*~IV5)])
    """

    terms = []
    text = line.strip()

    # Keep only the expression fragment for objective/constraint lines.
    if ":" in text:
        text = text.split(":", 1)[1]
    text = text.replace(";", " ").strip()
    if not text:
        raise ValueError(f"Could not parse OPB term from empty line: {line}")

    # OPB allows first term without an explicit sign. Normalize by prepending '+'
    # and then parse term-by-term using TERM_RE.
    if text and text[0] not in "+-":
        text = "+ " + text

    parsed_terms = TERM_RE.findall(text) # contains each part of the sum we're building
    if not parsed_terms:
        raise ValueError(f"Could not parse any OPB terms from line: {line}")

    for sign_tok, body in parsed_terms:
        parts = body.split()
        if not parts:
            raise ValueError(f"Missing variable token in OPB term: {line}")

        coeff = 1
        # Support unsigned/signed integer coefficients. Unsigned is interpreted as '+'.
        if parts[0].lstrip("+-").isdigit():
            coeff = int(parts[0])
            parts = parts[1:]
        if not parts:
            raise ValueError(f"Missing variable token in OPB term: {line}")

        factors = []
        for tok in parts:
            neg = tok.startswith("~")
            var_name = tok[1:] if neg else tok
            var_name = var_name.replace("\"", "") # support extended naming format
            if var_name not in vars:
                vars[var_name] = cp.boolvar(name=var_name)
            factors.append(~vars[var_name] if neg else vars[var_name])
        
        weight = -coeff if sign_tok == "-" else coeff
        term = weight * reduce(mul, factors, 1) # create weighted term
        terms.append(term)

    if len(terms) == 0:
        raise ValueError(f"Could not parse any OPB terms from line: {line}")

    return cp.sum(terms)

def _parse_constraint(line, vars):
    """
    Parse a single OPB constraint line into a CPMpy comparison expression.

    Arguments:
        line (str):                 A string representing a single OPB constraint.
        vars (list[cp.boolvar]):    List or array of CPMpy Boolean variables. Will be index to get the variables for the constraint.

    Returns:
        cp.expressions.core.Comparison: A CPMpy comparison expression representing
                                        the constraint.

    Example:
        >>> _parse_constraint("-1 x1 x14 -1 x1 ~x17 >= -1", vars)
        sum([-1, -1] * [(IV1*IV14), (IV1*~IV17)]) >= -1
    """

    match = IND_TERM_RE.search(line)
    if match is None:
        raise ValueError(f"Could not parse OPB comparator/rhs from line: {line}")
    op, ind_term = match.groups()
    lhs = _parse_term(line[:match.start()], vars)

    rhs = int(ind_term) if ind_term.lstrip("+-").isdigit() else vars[ind_term]

    return cp.expressions.core.Comparison(
        name="==" if op == "=" else ">=",
        left=lhs,
        right=rhs
    )

def _add_constraint(model: cp.Model, line: str, vars: dict[str, Any]):
    cons = _parse_constraint(line, vars)
    assert isinstance(cons, Comparison), cons
    lhs, rhs = cons.args
    assert isinstance(lhs, Operator) and lhs.name == "wsum" or lhs.name == "sum", lhs
    assert isinstance(rhs, int), rhs
    model.add(cons)

def load_opb(opb: Union[str, os.PathLike, TextIO], open:Callable = builtins.open) -> cp.Model:
    """
    Loader for OPB (Pseudo-Boolean) format. Loads an instance and returns its matching CPMpy model.

    Based on PyPBLib's example parser: https://hardlog.udl.cat/static/doc/pypblib/html/library/index.html#example-from-opb-to-cnf-file

    Supports:
        - Linear and non-linear terms (e.g., -1 x1 x14 +2 x2)
        - Negated variables using '~' (e.g., ~x5)
        - Minimisation objective
        - Comparison operators in constraints: '=', '>='

    Arguments:
        opb (str or os.PathLike or TextIO): 
            - A file path to an OPB file (optionally LZMA-compressed with `.xz`)
            - OR a string containing the OPB content directly
            - OR a TextIO object already open for reading
        open (Callable): callable to open the file for reading (default: builtin ``open``).
            Use for decompression, e.g. ``lambda p: lzma.open(p, 'rt')`` for ``.opb.xz``.

    Returns:
        cp.Model: The CPMpy model of the OPB instance.

    Example:
        >>> opb_text = '''
        ... * #variable= 5 #constraint= 2 #equal= 1 intsize= 64 #product= 5 sizeproduct= 13
        ... min: 2 x2 x3 +3 x4 ~x5 +2 ~x1 x2 +3 ~x1 x2 x3 ~x4 ~x5 ;
        ... 2 x2 x3 -1 x1 ~x3 = 5 ;
        ... '''
        >>> model = read_opb(opb_text)
        >>> print(model)
        Model(...)
    
    Notes:
        - Comment lines starting with '*' are ignored.
        - Only "min:" objectives are supported; "max:" is not recognized.
    """
    with _handle_loader_input(opb, open=open) as f:
        # Look for header on first line
        line = f.readline()
        header = HEADER_RE.match(line)
        if not header: # If not found on first line, look on second (happens when passing multi line string)
            _line = f.readline()
            header = HEADER_RE.match(_line)
            if not header:
                raise ValueError(f"Missing or incorrect header: \n0: {line}1: {_line}2: ...")
        nr_vars_declared = int(header.group(2))
        nr_constraints_declared = int(header.group(3))

        # Generator without comment lines
        reader = (line_text for line_text in map(str.strip, f) if line_text and line_text[0] != '*')

        # CPMpy objects
        vars: dict[str, Any] = {}
        model = cp.Model()

        with _ignore_strict_variable_name_check():
            # Special case for first line -> might contain objective function
            first_line = next(reader)
            if OBJ_TERM_RE.match(first_line):
                obj_expr = _parse_term(first_line, vars)
                model.minimize(obj_expr)
            else: # no objective found, parse as a constraint instead
                _add_constraint(model, first_line, vars)

            # Start parsing line by line
            for line in reader:
                _add_constraint(model, line, vars)
        _update_variable_counters(model)

        if len(vars) != nr_vars_declared:
            raise ValueError(f"Header declares {nr_vars_declared} variables but found {len(vars)} unique variables while parsing")
        
        if len(model.constraints) != nr_constraints_declared:
            raise ValueError(f"Header declares {nr_constraints_declared} constraints but found {len(model.constraints)} constraints while parsing")

        return model

# Non-extended (unquoted) VeriPB identifiers:
#  - start with an underscore or an ASCII letter (A-Z, a-z);
#  - continue with A-Z, a-z, 0-9, or []{}_^ (square/curly brackets, underscore, caret);
#  - be at least two characters long (and contain no spaces).
_VERIPB_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9\[\]{}_^]+\Z")

def _check_veripb_name(name: str) -> None:
    """
    Validate a variable name for use as an unquoted VeriPB identifier.

    Raises:
        ValueError: if ``name`` is not a valid VeriPB identifier.
    """
    if not _VERIPB_NAME_RE.match(name):
        raise ValueError(
            "A variable name for the OPB output is not a valid VeriPB identifier "
            f"(e.g. {name!r}): it must start with an underscore or a letter, continue "
            "with A-Z, a-z, 0-9, or []{}_^, and be at least two characters long. "
            "Rename the offending variable, use a different annotator, or "
            'switch to naming="restricted" to write competition-style x1, x2, ... '
            "names (with the original names preserved in comments)."
        )

def _wsum_to_str(cpm_expr, varmap):
    """
    Convert a weighted sum CPMpy expression to a string in OPB format.

    args:
        cpm_expr (Operator): wsum CPMpy expression
        varmap (dict): dictionary mapping CPMpy variables to OPB variable names
    """
    assert isinstance(cpm_expr, Operator) and cpm_expr.name == "wsum", f"Expected a wsum, but got {cpm_expr}"
    weights, args = cpm_expr.args

    out = []
    for w, var in zip(weights, args):
        if w == 0:
            continue
        lit = varmap[var] if not isinstance(var, NegBoolView) else "~" + str(varmap[var._bv])
        # OPB requires no space between sign and digits (e.g., "-2", "+4").
        out.append(f"{w:+d} {lit}")
    
    str_out = " ".join(out)
    return str_out

def write_opb(
        model:cp.Model, 
        path:Optional[Union[str, os.PathLike]] = None, 
        encoding:str="auto", 
        header:Optional[str] = None, 
        open:Callable = partial(builtins.open, mode="w"),
        annotate_bool: Optional[BooleanEncodingAnnotator] = None,
        naming: str = "restricted"
    ) -> str:
    """
    Export a CPMpy model to the OPB (Pseudo-Boolean) format.

    This function transforms the given CPMpy model into OPB format, which is a standard textual
    format for representing Pseudo-Boolean (optimization) problems. The OPB file will contain
    a header specifying the number of variables and constraints, an objective function (optional), 
    and a list of constraints using integer-weighted Boolean variables.

    Arguments:
        model (cp.Model): The CPMpy model to export.
        path (str or os.PathLike, optional): The file path to write the OPB output to.
        encoding (str, optional): The encoding used for `int2bool`. Options: ("auto", "direct", "order", "binary").
        header (str, optional): Optional header text to add as OPB comments.
            If None, a default CPMpy header is created only when writing to ``path``.
            Pass an empty string to skip adding a header.
        open (callable): Callable to open the file for writing (default: builtin ``open``).
            Called as ``open(path, "w")``. This mirrors the ``open=`` argument
            in loaders and allows custom compression or I/O (e.g.
            ``lambda p, mode='w': lzma.open(p, 'wt')``).
        annotate_bool (BooleanEncodingAnnotator, optional): encoding annotator, annotates 
            boolean variables with names describing how they contribute to an encoded integer variable.
            Depending on the `naming` scheme, these names are written as comments or used directly as 
            variable names.
        naming (str): how variables are named in the OPB output. One of:

            - ``"restricted"`` (default): competition-style ``x1``, ``x2``, ... identifiers. 
              Annotations, if provided, are written as comments.
            - ``"extended"``: allows arbitrary variable names, but wraps them in quotes.
            - ``"veripb"``: VeriPB-safe variable names.

    .. note::
        The ``"restricted"`` naming scheme is the most widely supported and 
        compatible with most OPB parsers. The ``"extended"`` naming scheme is 
        more flexible and allows arbitrary characters, but is not as widely 
        supported. The ``"veripb"`` naming scheme follows the VeriPB convention;

        - start with an underscore or an ASCII letter (A-Z, a-z);
        - continue with A-Z, a-z, 0-9, or []{}_^ (square/curly brackets, underscore, caret);
        - be at least two characters long 
        - contain no spaces

        Any variable name that does not follow these rules will be rejected.
            

    Returns:
        str: The OPB string (as it is optionally written to a file).

    Format:
        * #variable= <n_vars> #constraint= <n_constraints>
        min/max: <objective>;
        <constraint_1>;
        <constraint_2>;
        ...

    Example:
        >>> from cpmpy import *
        >>> x = boolvar(shape=3)
        >>> m = Model(x[0] + x[1] + x[2] >= 2)
        >>> print(write_opb(m))
    """

    if header is None:
        header = _create_header(format="opb") if path is not None else None
    elif header == "":
        header = None

    csemap, ivarmap = CSEMap(), dict[str, Any]()
    opb_cons = to_opb(model.constraints, csemap, ivarmap, encoding)

    # Transform objective, if present
    if model.objective_ is not None:
        opb_obj, const, extra_cons = to_opb_objective(model.objective_, csemap, ivarmap, encoding)
        opb_cons += to_opb(extra_cons, csemap, ivarmap, encoding)
    else:
        opb_obj = None

    # Form header and variable mapping
    # Use all variables occurring in constraints and the objective
    all_vars = get_variables(opb_cons + ([opb_obj] if opb_obj is not None else []))
    out = [
        f"* #variable= {len(all_vars)} #constraint= {len(opb_cons)}",
    ]
    if header:
        header_lines = ["* " + line for line in str(header).splitlines()]
        out.extend(header_lines)

    # Descriptive names from the encoding annotator (or None if no annotator)
    names = annotate_bool.annotate(all_vars, ivarmap) if annotate_bool is not None else None

    # Map variables to OPB variable names according to the chosen naming scheme
    if naming == "restricted":
        # Competition-style x<i> identifiers; descriptive names go to comments.
        varmap = {v: f"x{i+1}" for i, v in enumerate(all_vars)}
        if names is not None:
            out.extend(f"* x{i+1} {names[i]}" for i in range(len(all_vars)))
    else:
        base = names if names is not None else [v.name for v in all_vars]
        if naming == "extended":
            # Quoted names allow arbitrary characters.
            varmap = {v: f'"{base[i]}"' for i, v in enumerate(all_vars)}
        elif naming == "veripb":
            # Unquoted names; characters must be valid VeriPB identifiers.
            for nm in base:
                _check_veripb_name(nm)
            varmap = {v: base[i] for i, v in enumerate(all_vars)}
        else:
            raise ValueError(f"Unknown naming {naming!r}; choose 'restricted', 'extended', or 'veripb'")

    # Write objective, if present
    if model.objective_ is not None:
        objective_str = _wsum_to_str(opb_obj, varmap)
        out.append(f"{'min' if model.objective_is_min else 'max'}: {objective_str};")

    # Write constraints
    for cons in opb_cons:
        assert isinstance(cons, Comparison), f"Expected a comparison, but got {cons}"
        lhs, rhs = cons.args
        constraint_str = f"{_wsum_to_str(lhs, varmap)} {cons.name} {rhs};"
        out.append(constraint_str)

    # Output to file or string
    contents = "\n".join(out)
    if path is not None:
        with open(path) as f:
            f.write(contents)
        
    return contents
