#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## scip.py
##
"""
This file implements helper functions for converting CPMpy models to and from various data 
formats supported by the SCIP optimization suite.

============
Installation
============

The 'pyscipopt' optional dependency must be installed separately through `pip`:

.. code-block:: console
    
    $ pip install cpmpy[io.scip]

=================
List of functions
=================

.. autosummary::
    :nosignatures:

    load_scip
    write_scip
"""


import math
import os
import tempfile
from io import TextIOBase
import cpmpy as cp
import warnings
import builtins
from typing import Union, Optional, Callable, TYPE_CHECKING, TextIO
from functools import partial

if TYPE_CHECKING:
    import pyscipopt

from cpmpy.solvers.scip import CPM_scip
from cpmpy.expressions.variables import _ignore_strict_variable_name_check
from cpmpy.model import _update_variable_counters
from cpmpy.tools.io.utils import _create_header, _derive_format, get_extension


def load_scip(instance: Union[str, os.PathLike, TextIO], open:Callable = builtins.open, assume_integer:bool=False, type: Optional[str]=None) -> cp.Model:
    """
    Load a SCIP-compatible model from a file and return a CPMpy model.

    Arguments:
        instance (str or os.PathLike or TextIO): The path to the SCIP-compatible file to read, a string containing the model content directly, or a TextIO object already open for reading.
        open (Callable): The function to use to open the file. (SCIP does not require this argument, will be ignored)
        assume_integer (bool): Whether to assume that all variables are integer.
        type (str, optional): Type of the inline string or unnamed TextIO input. Required when ``instance`` is a raw string.

    Warning:
        Setting assumes_integer to True will cause CPMpy to assume that all variables are integer, 
        even if they are not explicitly declared as such in the problem file. Use with caution.

    Returns:
        cp.Model: A CPMpy model.
    """

    # Check if SCIP is installed
    if not _SCIPWriter.supported():
        raise Exception("SCIP: Install SCIP IO dependencies: cpmpy[io.scip]")

    from pyscipopt import Model

    content = None
    if isinstance(instance, TextIOBase):
        content = instance.read()
        if type is None:
            stream_name = getattr(instance, "name", None)
            if not isinstance(stream_name, (str, os.PathLike)):
                raise ValueError("Type must be provided when loading a SCIP model from an unnamed TextIO object.")
            type = _derive_format(stream_name)
      
    elif isinstance(instance, str) and not os.path.exists(instance):
        if type is None:
            raise ValueError("Type must be provided when loading a SCIP model from a string.")
        content = instance

    # SCIP's parser only supports file paths
    # -> write content to a temporary file
    tmp_fname = None
    if content is not None:
        assert type is not None
        try:
            suffix = "." + get_extension(type)
        except KeyError as e:
            raise ValueError(f"Unsupported SCIP file type: {type}") from e

        with tempfile.NamedTemporaryFile(suffix=suffix, mode="w", delete=False) as tmp:
            tmp.write(content)
            tmp_fname = tmp.name
        instance = tmp_fname

    # Load file into pyscipopt model
    scip = Model()
    try:
        scip.hideOutput() # suppress SCIP output
        scip.readProblem(filename=instance)
        scip.hideOutput(quiet=False)
    finally:
        if tmp_fname is not None:
            os.remove(tmp_fname)

    # 1) translate variables
    scip_vars = scip.getVars()
    var_map = {}
    with _ignore_strict_variable_name_check():
        for var in scip_vars:
            name = var.name         # name of the variable
            vtype = var.vtype()     # type of the variable
            if vtype == "BINARY":
                var_map[name] = cp.boolvar(name=name)
            elif vtype == "INTEGER":
                lb = int(var.getLbOriginal())
                ub = int(var.getUbOriginal())
                var_map[name] = cp.intvar(lb, ub, name=name)
            elif vtype == "CONTINUOUS":
                if assume_integer:
                    lb = int(math.ceil(var.getLbOriginal()))
                    ub = int(math.floor(var.getUbOriginal()))
                    if lb != var.getLbOriginal() or ub != var.getUbOriginal():
                        warnings.warn(f"Continuous variable {name} has non-integer bounds {var.getLbOriginal()} - {var.getUbOriginal()}. CPMpy will assume it is integer.")
                    var_map[name] = cp.intvar(lb, ub, name=name)
                else:
                    raise ValueError(f"CPMpy does not support continious variables: {name}")
            else:
                raise ValueError(f"Unsupported variable type: {vtype}")
        

    model = cp.Model()

    # 2) translate constraints
    scip_cons = scip.getConss()
    for cons in scip_cons:
        ctype = cons.getConshdlrName()  # type of the constraint

        if ctype == "linear":
            cons_vars = scip.getConsVars(cons)  # variables in the constraint (x)
            cons_coeff = scip.getConsVals(cons) # coefficients of the variables (A)

            cpm_vars = [var_map[v.name] for v in cons_vars] # convert to CPMpy variables
            cpm_sum = cp.sum(var*coeff for (var,coeff) in zip(cpm_vars, cons_coeff)) # Ax

            lhs = scip.getLhs(cons) # lower bound of the constraint (-infinity if one-sided)
            rhs = scip.getRhs(cons) # upper bound of the constraint (+infinity if one-sided)

            # SCIP encodes an open side with +/- infinity (a large sentinel, e.g. 1e20).
            # Such a side is not a real bound, so it must not be turned into a constraint
            # (materialising it would also overflow solver integer limits).
            has_lhs = not scip.isInfinity(-lhs)
            has_rhs = not scip.isInfinity(rhs)

            def _check_integer(bound, rounded):
                if rounded != bound:
                    if assume_integer:
                        warnings.warn(f"Constraint {cons.name} has non-integer bounds. CPMpy will assume it is integer.")
                    else:
                        raise ValueError(f"Constraint {cons.name} has non-integer bounds. CPMpy does not support non-integer bounds.")

            # add the (finite) constraint bound(s) to the model
            if has_lhs:
                _lhs = math.ceil(lhs)
                _check_integer(lhs, _lhs)
                model += int(_lhs) <= cpm_sum
            if has_rhs:
                _rhs = math.floor(rhs)
                _check_integer(rhs, _rhs)
                model += cpm_sum <= int(_rhs)

        else: 
            raise ValueError(f"Unsupported constraint type: {ctype}")

    # 3) translate objective
    scip_objective = scip.getObjective()
    direction = scip.getObjectiveSense()

    objective = _load_scip_objective(scip_objective, var_map, assume_integer)

    if direction == "minimize":
        model.minimize(objective)
    elif direction == "maximize":
        model.maximize(objective)
    else:
        raise ValueError(f"Unsupported objective sense: {direction}")

    _update_variable_counters(model)
    return model

def _load_scip_objective(scip_objective, var_map, assume_integer: bool):
    """
    Translate a SCIP objective to a CPMpy objective.

    Arguments:
        scip_objective: The SCIP objective to translate.
        var_map: A dictionary mapping SCIP variable names to CPMpy variables.
        assume_integer: Whether to assume that all variables are integer.

    Returns:
        The CPMpy objective.

    Raises:
        ValueError: If the objective term has a non-integer coefficient and assume_integer is False.
    """
    obj_terms = []

    for term, coeff in scip_objective.terms.items(): # terms is a dictionary mapping terms to coefficients
        _coeff = int(math.floor(coeff))
        if _coeff != int(coeff):
            if assume_integer:
                warnings.warn(f"Objective term {term} has non-integer coefficient. CPMpy will assume it is integer.")
            else:
                raise ValueError(f"Objective term {term} has non-integer coefficient. CPMpy does not support non-integer coefficients.")

        cpm_term = 1
        for scip_var in term.vartuple:
            cpm_term *= var_map[scip_var.name]
        obj_terms.append(_coeff * cpm_term)

    return cp.sum(obj_terms)

class _SCIPWriter(CPM_scip):
    """
    A helper class aiding in translating CPMpy models to SCIP models.

    Builds on top of the CPMpy SCIP solver interface.
    """
    def __init__(self, model: cp.Model, problem_name: Optional[str] = None):
        if not self.supported():
            raise Exception(
                "SCIP: Install SCIP IO dependencies: cpmpy[io.scip]")

        super().__init__(model)
        self.scip_model.setProbName(problem_name)

def _add_header(path: Union[str, os.PathLike], format: str, header: Optional[str] = None):
    """
    Add a header to a file.

    Arguments:
        path (str or os.PathLike): The path to the file to add the header to.
        format (str): The format of the file.
        header (Optional[str]): The header to add.
    """

    if header is None:
        header = ""

    with open(path, "r") as f:
        lines = f.readlines()

    if format == "mps":
        header_lines = ["* " + line + "\n" for line in header.splitlines()]
        lines = header_lines + lines
        
    elif format == "lp":
        header_lines = ["\\ " + line + "\n" for line in header.splitlines()]
        lines = header_lines + lines

    elif format == "cip":
        header_lines = ["# " + line + "\n" for line in header.splitlines()]
        lines = header_lines + lines

    elif format == "fzn":
        header_lines = ["% " + line + "\n" for line in header.splitlines()]
        lines = header_lines + lines

    elif format == "gms":
        header_lines = ["* " + line + "\n" for line in header.splitlines()]
        lines = [lines[0]] + header_lines + lines[1:] # handle first line: $OFFLISTING

    elif format == "pip":
        header_lines = ["\\ " + line + "\n" for line in header.splitlines()]
        lines = header_lines + lines

    else:
        warnings.warn(f"Unsupported format for header: {format}")
        return

    with open(path, "w") as f:
        f.writelines(lines)


def write_scip(
        model: cp.Model, 
        path: Optional[Union[str, os.PathLike]] = None, 
        format: str = "mps", 
        header: Optional[str] = None, 
        verbose: bool = False, 
        open: Callable = partial(builtins.open, mode="w")
    ) -> str:
    """
    Write a CPMpy model to file using the SCIP solver.

    Supported formats include: 
    - "mps"
    - "lp"
    - "cip"
    - "fzn"
    - "gms"
    - "pip"

    More formats can be supported upon the installation of additional dependencies (like SIMPL).
    For more information, see the SCIP documentation: https://pyscipopt.readthedocs.io/en/latest/tutorials/readwrite.html

    Arguments:
        model (cp.Model): CPMpy model to write.
        path (str or os.PathLike, optional): The file path to write the SCIP output to. If None, the SCIP string is returned.
        format (str): Output format (e.g. "mps", "lp", "cip", "fzn", "gms", "pip").
        header (Optional[str]): Optional header text to prepend (format-dependent comment style).
            If None, a default CPMpy header is created only when writing to ``path``.
            Pass an empty string to skip adding a header.
        verbose (bool): If True, allow SCIP to print progress.
        open (Callable): Callable to open the file for writing (default: builtin ``open``).
            Called as ``open(path)``. Mirrors the ``open=`` argument in loaders and
            allows custom compression or I/O (e.g.
            ``lambda p: lzma.open(p, 'wt')``).

    Returns:
        str: The file content as a string (whether written to ``path`` or not).
    """

    writer = _SCIPWriter(model, problem_name="CPMpy Model")
    if header is None:
        header = _create_header(format=format) if path is not None else None
    elif header == "":
        header = None

    # Always write via SCIP to a temp file, then add header and get content
    with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as tmp:
        tmp_fname = tmp.name
    try:
        if not verbose:
            writer.scip_model.hideOutput()
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stdout = os.dup(1)
        os.dup2(devnull, 1)
        try:
            writer.scip_model.writeProblem(tmp_fname, verbose=verbose)
        finally:
            os.dup2(old_stdout, 1)
            os.close(devnull)
            os.close(old_stdout)
        if not verbose:
            writer.scip_model.hideOutput(quiet=False)
        _add_header(tmp_fname, format, header)
        with builtins.open(tmp_fname, "r") as f:
            content = f.read()
        if path is not None:
            with open(path) as f:
                f.write(content)
        return content
    finally:
        os.remove(tmp_fname)
