"""
Annotate Boolean encoding variables.

Some target formats don't support integer variables and thus require them to be encoded into a set of 
Boolean variables. For example, a direct (or one-hot) encoding of an integer variable `x` in the range 
`[0, 10]` requires 11 Boolean variables, one for each possible value of `x`. 

When writing to file, this mapping between the integer variable and the Boolean variables is typically lost. 
If we also save an annotation to the file, a mapping from Boolean variable name to integer variable name, 
then an external tool can recover it later.

Additionally, some formats don't support arbitrary variable naming. In DIMACS for example, 
variables are referenced by integer IDs. Storing an annotation in the file allows again 
to map it back later if needed.

===============
List of classes
===============

.. autosummary::
    :nosignatures:

    BooleanEncodingAnnotator
    SugarAnnotator
    VeriPBAnnotator
"""

from abc import ABC, abstractmethod
from typing import Any

from cpmpy.transformations.int2bool import IntVarEnc, IntVarEncDirect, IntVarEncOrder, IntVarEncLog


class BooleanEncodingAnnotator(ABC):
    """
    Abstract base class for encoding annotators.

    An annotator maps each Boolean variable to a single descriptive name: either
    the variable's own name (for variables that were already Boolean) or a name
    that captures how the Boolean encodes an integer variable (e.g. ``x_ge_2``).

    How that name is placed in a concrete file format (as a comment, as the
    variable identifier itself, quoted, ...) is decided by the writer, not here.
    """
    @abstractmethod
    def annotate(self, vars: list[Any], ivarmap: dict[str, IntVarEnc]) -> list[str]:
        """Return one name per variable, in the same order as ``vars``."""
        pass

class SugarAnnotator(BooleanEncodingAnnotator):
    """
    Return Sugar-style annotations for Boolean encoding variables.

    Sugar-style convention:
      order:  p<var>,<a>      meaning <var> <= a
      direct: p<var>=<a>      meaning <var> = a
      binary: p<var>#<i>      meaning bit i of <var>

    
    .. note::

        Returned strings are typically written as DIMACS comments:
        ``c <lit_id> <name>``, but this depends on the targeted output format.

    Arguments:
        vars: Boolean encoding variables.
        ivarmap: Integer encoding map populated by the CNF/DIMACS
            transformation.

    Returns:
        Sugar-style names in the same order as ``vars``.

    Example:
        With variables ``[b, BV0, BV1, BV2, BV4, BV5, BV6]`` where:
        - ``BV0`` and ``BV1`` are order literals for ``x in 1..3``, 
        - ``BV2`` and ``BV4`` are direct literals for ``y in 0..2``, and 
        - ``BV5`` and ``BV6`` are binary bits for ``z in 4..7``:

        >>> SugarAnnotator().annotate(vars, ivarmap)
        ['b', 'px,2', 'px,3', 'py=0', 'py=2', 'pz#0', 'pz#1']
    """

    def annotate(self, vars: list[Any], ivarmap: dict[str, IntVarEnc]) -> list[str]:
        """
        Return Sugar-style names for Boolean encoding variables.

        Arguments:
            vars: Boolean encoding variables.
            ivarmap: Integer encoding map populated by the int2bool transformation.

        Returns:
            Sugar-style names in the same order as ``vars``.
        """       

        reverse = _build_reverse_map(ivarmap)
        lines = []

        for v in vars:
            info = reverse.get(id(v))

            if info is None:
                lines.append(_safe_name(v))
                continue

            src = info["source_name"]

            if info["encoding"] == "order":
                # Sugar order encoding: px,a means x <= a
                name = f"p{src},{info['threshold']}"

            elif info["encoding"] == "direct":
                # Sugar-like direct encoding: px=a means x = a
                name = f"p{src}={info['value']}"

            elif info["encoding"] == "binary":
                # Sugar-like binary/log encoding: px#i means bit i of x
                name = f"p{src}#{info['bit']}"

            else:
                name = _safe_name(v)

            lines.append(name)

        return lines

class VeriPBAnnotator(BooleanEncodingAnnotator):
    """
    Return VeriPB-safe annotations for Boolean encoding variables.

    Integer encodings are mapped back to underscore-separated names:
    order literals become ``x_ge_value``, direct literals become
    ``x_eq_value``, and binary/log literals become ``x_bit<i>``. Commas in
    variable names are replaced because VeriPB does not accept them.
    Unencoded CPMpy auxiliary variables whose names start with ``BV`` are
    prefixed with ``_``, as per convention.

    Arguments:
        vars: Boolean encoding variables.
        ivarmap: Integer encoding map populated by the CNF/DIMACS
            transformation.

    Returns:
        VeriPB-safe names in the same order as ``vars``.

    Example:
        With variables ``[b, BV0, BV1, BV2, BV4, BV5, BV6]`` where:
        - ``BV0`` and ``BV1`` are order literals for ``x in 1..3``, 
        - ``BV2`` and ``BV4`` are direct literals for ``y in 0..2``, and 
        - ``BV5`` and ``BV6`` are binary bits for ``z in 4..7``:

        >>> VeriPBAnnotator().annotate(vars, ivarmap)
        ['b', 'x_ge_2', 'x_ge_3', 'y_eq_0', 'y_eq_2', 'z_bit0', 'z_bit1']
    """

    def annotate(self, vars: list[Any], ivarmap: dict[str, IntVarEnc]) -> list[str]:
        """
        Return VeriPB-safe names for Boolean encoding variables.

        Arguments:
            vars: Boolean encoding variables.
            ivarmap: Integer encoding map populated by the int2bool transformation.

        Returns:
            VeriPB-safe names in the same order as ``vars``.
        """
        
        reverse = _build_reverse_map(ivarmap)
        names = []
        for v in vars:
            info = reverse.get(id(v))
            if info is None:
                vname = _safe_name(v)
                if str(vname).startswith("BV"): # aux vars introduced by CPMpy
                    names.append("_" + _veripb_safe_name(vname))
                else:
                    names.append(_veripb_safe_name(vname))
            elif info["encoding"] == "order":
                names.append(f"{_veripb_safe_name(info['source_name'])}_ge_{info['threshold']}")
            elif info["encoding"] == "binary":
                names.append(f"{_veripb_safe_name(info['source_name'])}_bit{info['bit']}")
            elif info["encoding"] == "direct":
                names.append(f"{_veripb_safe_name(info['source_name'])}_eq_{info['value']}")
            else:
                names.append(_veripb_safe_name(_safe_name(v)))
        return names

def _safe_name(v: Any) -> str:
    """Best-effort variable label for unexpected objects (e.g. lists)."""
    return getattr(v, "name", str(v))

def _veripb_safe_name(name: str) -> str:
    """VeriPB does not support commas in variable names."""
    s = str(name)
    return s.replace(", ", "][").replace(",", "][")

def _build_reverse_map(ivarmap: dict[str, IntVarEnc]) -> dict[int, dict[str, Any]]:
    """
    Build a reverse lookup from BoolVar object ID to decode info.

    Returns: dict {id(BoolVar) -> {"source_name": str, "encoding": str, ...}}
    """
    reverse = {}
    for _var_name, enc in ivarmap.items():
        orig_name = enc._x.name
        if isinstance(enc, IntVarEncOrder):
            # xs[i] represents the condition  x >= lb + 1 + i
            for i, bv in enumerate(enc._xs):
                reverse[id(bv)] = {
                    "source_name": orig_name,
                    "encoding": "order",
                    "threshold": enc._x.lb + 1 + i,
                }
        elif isinstance(enc, IntVarEncDirect):
            # xs[i] represents the condition  x == lb + i
            for i, bv in enumerate(enc._xs):
                reverse[id(bv)] = {
                    "source_name": orig_name,
                    "encoding": "direct",
                    "value": enc._x.lb + i,
                }
        elif isinstance(enc, IntVarEncLog):
            # xs[i] is the bit at position i (value contribution: 2^i, offset by lb)
            for i, bv in enumerate(enc._xs):
                reverse[id(bv)] = {
                    "source_name": orig_name,
                    "encoding": "binary",
                    "bit": i,
                }
    return reverse

    
