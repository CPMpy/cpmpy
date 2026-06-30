from typing import Any, Callable

from cpmpy.transformations.int2bool import IntVarEnc, IntVarEncDirect, IntVarEncOrder, IntVarEncLog


AnnotationCallable = Callable[[list[Any], dict[str, IntVarEnc]], list[str]]
"""Callable shape for writer annotation hooks: ``annotate(vars, ivarmap) -> list[str]``."""


def _safe_name(v):
    """Best-effort variable label for unexpected objects (e.g. lists)."""
    return getattr(v, "name", str(v))

def _veripb_safe_name(name):
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

def annotate_cpmpy(vars: list[Any], ivarmap: dict[str, IntVarEnc]) -> list[str]:
    """
    Return CPMpy-style variable annotations for Boolean encoding variables.

    CPMpy-like naming convention:
    - order literals: ``x>=value``
    - direct literals: ``x=value``
    - binary/log literals: ``x[bit=i]``
    - unencoded Booleans: their original variable name

    Arguments:
        vars: Boolean encoding variables.
        ivarmap: Integer encoding map populated by the CNF/DIMACS transformation.

    Returns:
        Variable names in the same order as ``vars``.

    Example:
        With variables ``[b, BV0, BV1, BV2, BV4, BV5, BV6]`` where:
        - ``BV0`` and ``BV1`` are order literals for ``x in 1..3``, 
        - ``BV2`` and ``BV4`` are direct literals for ``y in 0..2``, and 
        - ``BV5`` and ``BV6`` are binary bits for ``z in 4..7``:

        >>> annotate_cpmpy(vars, ivarmap)
        ['b', 'x>=2', 'x>=3', 'y=0', 'y=2', 'z[bit=0]', 'z[bit=1]']
    """
    reverse = _build_reverse_map(ivarmap)
    lines = []
    for v in vars:
        info = reverse.get(id(v))
        if info is None:
            name = _safe_name(v)
        elif info["encoding"] == "order":
            name = f"{info['source_name']}>={info['threshold']}"
        elif info["encoding"] == "binary":
            name = f"{info['source_name']}[bit={info['bit']}]"
        elif info["encoding"] == "direct":
            name = f"{info['source_name']}={info['value']}"
        else:
            name = _safe_name(v)
        lines.append(name)
    return lines

def annotate_sugar(vars: list[Any], ivarmap: dict[str, IntVarEnc]) -> list[str]:
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

        >>> annotate_sugar(vars, ivarmap)
        ['b', 'px,2', 'px,3', 'py=0', 'py=2', 'pz#0', 'pz#1']
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

def annotate_veripb(vars: list[Any], ivarmap: dict[str, IntVarEnc]) -> list[str]:
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

        >>> annotate_veripb(vars, ivarmap)
        ['b', 'x_ge_2', 'x_ge_3', 'y_eq_0', 'y_eq_2', 'z_bit0', 'z_bit1']
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
