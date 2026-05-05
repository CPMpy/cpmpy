from cpmpy.transformations.int2bool import IntVarEncDirect, IntVarEncOrder, IntVarEncLog

def _build_reverse_map(ivarmap):
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

def annotate_cpmpy(vars, ivarmap):
    reverse = _build_reverse_map(ivarmap)
    lines = []
    for v in vars:
        info = reverse.get(id(v))
        if info is None:
            name = v.name
        elif info["encoding"] == "order":
            name = f"{info['source_name']}>={info['threshold']}"
        elif info["encoding"] == "binary":
            name = f"{info['source_name']}[bit={info['bit']}]"
        elif info["encoding"] == "direct":
            name = f"{info['source_name']}={info['value']}"
        else:
            name = v.name
        lines.append(name)
    return lines

# def annotate_veripb(vars, ivarmap):
#     """
#     Underscore-style encoding annotation, following VeriPB
#     """
#     reverse = _build_reverse_map(ivarmap)
#     lines = []
#     for i,v in enumerate(vars):
#         lit_id = i+1
#         info = reverse.get(id(v))
#         if info is None:
#             continue
#         elif info["encoding"] == "order":
#             name = f"{info['source_name']}_ge_{info['threshold']}"
#         elif info["encoding"] == "direct":
#             name = f"{info['source_name']}_eq_{info['value']}"
# #        elif info["encoding"] == "binary":
# #            name = f"{info['source_name']}_bit{info['bit']}"
#         elif info["encoding"] == "binary":
#             # Using dot notation for sign/magnitude vectors
#             name = f"{info['source_name']}.m{info['bit']}"
#         else:
#             name = v.name
#         lines.append(f"{lit_id} {name}")
#     return lines

def annotate_sugar(vars, ivarmap):
    """
    Sugar-style encoding annotation.

    Sugar-style convention:
      order:  p<var>,<a>      meaning <var> <= a
      direct: p<var>=<a>      meaning <var> = a
      binary: p<var>#<i>      meaning bit i of <var>

    Returned lines become DIMACS comments:
      c <lit_id> <name>
    """
    reverse = _build_reverse_map(ivarmap)
    lines = []

    for v in vars:
        info = reverse.get(id(v))

        if info is None:
            name = v.name

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
            name = v.name

        lines.append(name)

    return lines

def annotate_veripb(vars, ivarmap):
    reverse = _build_reverse_map(ivarmap)
    names = []
    for v in vars:
        info = reverse.get(id(v))
        if info is None:
            if v.name[:2] == "BV": # aux vars introduced by CPMpy
                names.append("_" + v.name)
        elif info["encoding"] == "order":
            names.append(f"{info['source_name']}_ge_{info['threshold']}")
        elif info["encoding"] == "binary":
            names.append(f"{info['source_name']}_bit{info['bit']}")
        elif info["encoding"] == "direct":
            names.append(f"{info['source_name']}_eq_{info['value']}")
        else:
            names.append(v.name)
    return names


