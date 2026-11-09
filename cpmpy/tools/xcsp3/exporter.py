"""Write CPMpy models to XCSP3 XML.

Unsupported constructs fail before writing the destination.
"""
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
from cpmpy.transformations.get_variables import get_variables

_SIMPLE_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_BINARY = {"==": "eq", "!=": "ne", "<": "lt", "<=": "le", ">": "gt", ">=": "ge",
           "sub": "sub", "mul": "mul", "div": "div", "mod": "mod", "pow": "pow",
           "->": "imp", "and": "and", "or": "or", "xor": "xor"}
_NARY = {"sum": "add", "min": "min", "max": "max"}
_UNARY = {"abs": "abs", "-": "neg", "not": "not", "~": "not"}
_GLOBAL = {"alldifferent": "allDifferent", "allequal": "allEqual"}


def _is_variable(expr):
    return (hasattr(expr, "lb") and hasattr(expr, "ub")
            and hasattr(expr, "name") and not hasattr(expr, "args"))


def _identifier_map(variables):
    """Assign unique simple XCSP3 IDs, including to CPMpy array elements."""
    result = {}
    used = set()
    for i, var in enumerate(variables):
        name = str(var.name)
        # Preserve ordinary names where possible; rename indexed and invalid names.
        candidate = name if _SIMPLE_ID.fullmatch(name) else f"v_{i}"
        if candidate in used:
            candidate = f"v_{i}"
        while candidate in used:
            candidate += "_"
        used.add(candidate)
        result[id(var)] = candidate
    return result


class _VariableRegistry(dict):
    """Identifier map that also tracks variables introduced by decompositions."""

    def __init__(self, variables):
        super().__init__(_identifier_map(variables))
        self.variables = list(variables)

    def register(self, variables):
        known = set(self)
        used = set(self.values())
        for var in variables:
            if id(var) in known:
                continue
            candidate = str(var.name)
            if not _SIMPLE_ID.fullmatch(candidate) or candidate in used:
                candidate = f"aux_{len(self.variables)}"
            while candidate in used:
                candidate += "_"
            self[id(var)] = candidate
            self.variables.append(var)
            known.add(id(var))
            used.add(candidate)


def _expr(expr, names):
    if isinstance(expr, (bool, np.bool_)):
        return "1" if expr else "0"
    if isinstance(expr, (int, np.integer)):
        return str(int(expr))
    if _is_variable(expr):
        try:
            return names[id(expr)]
        except KeyError as exc:
            raise ValueError(f"Undeclared variable: {expr}") from exc

    name, args = getattr(expr, "name", None), getattr(expr, "args", None)
    if name is None or args is None:
        raise NotImplementedError(f"Unsupported XCSP3 expression: {type(expr).__name__}: {expr}")
    if name == "allequal":
        # The importer maps n-ary ne(...) to ~AllEqual(...).
        # AllEqual can therefore occur nested inside a Boolean expression.
        items = list(args)
        if len(items) < 2:
            return "1"
        terms = [f"eq({_expr(items[0], names)},{_expr(item, names)})"
                 for item in items[1:]]
        return terms[0] if len(terms) == 1 else f"and({','.join(terms)})"
    if name == "element":
        array, index = args
        array = list(array)
        if not array:
            raise ValueError("Element array cannot be empty")
        # An XCSP3 conditional expression has no out-of-range indexing.
        # Reject domains containing invalid indices rather than changing semantics.
        if not _is_variable(index) or index.lb < 0 or index.ub >= len(array):
            raise NotImplementedError("Element currently requires an index variable with an in-range domain")
        index_xml = _expr(index, names)
        result = _expr(array[-1], names)
        for i in reversed(range(len(array) - 1)):
            result = f"if(eq({index_xml},{i}),{_expr(array[i], names)},{result})"
        return result
    if name == "count":
        array, value = args
        terms = [f"eq({_expr(item, names)},{_expr(value, names)})" for item in array]
        return "add(" + ",".join(terms) + ")" if len(terms) > 1 else (terms[0] if terms else "0")
    if name == "among":
        array, values = args
        unique_values = list(dict.fromkeys(int(v) for v in values))
        terms = []
        for item in array:
            comparisons = [f"eq({_expr(item, names)},{v})" for v in unique_values]
            terms.append("or(" + ",".join(comparisons) + ")" if len(comparisons) > 1 else (comparisons[0] if comparisons else "0"))
        return "add(" + ",".join(terms) + ")" if len(terms) > 1 else (terms[0] if terms else "0")
    if name in {"nvalue", "nvalue_except"}:
        if name == "nvalue":
            sequence, excluded = list(args), None
        else:
            sequence, excluded = list(args[0]), int(args[1])
        # Count the first occurrence of each distinct value, avoiding
        # potentially huge enumerations of the variable domains.
        indicators = []
        for i, item in enumerate(sequence):
            term = _expr(item, names)
            earlier = [f"ne({term},{_expr(prev, names)})" for prev in sequence[:i]]
            if excluded is not None:
                earlier.insert(0, f"ne({term},{excluded})")
            indicators.append("and(" + ",".join(earlier) + ")" if len(earlier) > 1 else (earlier[0] if earlier else "1"))
        return "add(" + ",".join(indicators) + ")" if len(indicators) > 1 else (indicators[0] if indicators else "0")
    if name == "nd_element":
        arr, *indices = args
        if len(indices) != arr.ndim or any(
            not _is_variable(idx) or idx.lb < 0 or idx.ub >= size
            for idx, size in zip(indices, arr.shape)
        ):
            raise NotImplementedError("NDElement requires one in-range index variable per dimension")
        # Element with an expression index is not yet supported: expand
        # over all legal index combinations without adding auxiliaries.
        terms = []
        for coordinates in np.ndindex(arr.shape):
            guards = [f"eq({_expr(idx, names)},{coordinate})" for idx, coordinate in zip(indices, coordinates)]
            guard = guards[0] if len(guards) == 1 else f"and({','.join(guards)})"
            terms.append((guard, _expr(arr[coordinates], names)))
        result = terms[-1][1]
        for guard, value in reversed(terms[:-1]):
            result = f"if({guard},{value},{result})"
        return result
    if name == "indomain":
        value, domain = args
        values = sorted(set(int(v) for v in domain))
        if not values:
            return "0"
        terms = [f"eq({_expr(value, names)},{v})" for v in values]
        return terms[0] if len(terms) == 1 else f"or({','.join(terms)})"
    if name == "ite" and len(args) == 3:
        return "if(" + ",".join(_expr(a, names) for a in args) + ")"
    if name == "xor":
        # XCSP3 xor is Boolean parity, including n-ary operands.
        if not args:
            return "0"
        if len(args) == 1:
            return _expr(args[0], names)
        return "xor(" + ",".join(_expr(a, names) for a in args) + ")"
    if name == "dist":
        if len(args) != 2:
            raise NotImplementedError("dist requires exactly two operands")
        return f"dist({_expr(args[0], names)},{_expr(args[1], names)})"
    if name == "iff":
        # The importer accepts n-ary iff: every adjacent pair must agree.
        if not args:
            return "1"
        if len(args) == 1:
            return "1"
        if len(args) == 2:
            return f"iff({_expr(args[0], names)},{_expr(args[1], names)})"
        comparisons = [f"eq({_expr(a, names)},{_expr(b, names)})"
                       for a, b in zip(args[:-1], args[1:])]
        return f"and({','.join(comparisons)})"
    if name in {"in", "notin"}:
        if len(args) != 2:
            raise NotImplementedError(f"{name} requires a value and a finite set")
        value, domain = args
        if not isinstance(domain, (list, tuple, set, frozenset, np.ndarray)):
            raise NotImplementedError(f"{name} requires an explicit finite set")
        if not all(isinstance(v, (int, np.integer)) for v in domain):
            raise NotImplementedError(f"{name} requires integer set elements")
        values = sorted(set(map(int, domain)))
        if not values:
            return "0" if name == "in" else "1"
        # Some supported pycsp3/CPMpy importer combinations cannot evaluate
        # multi-element notin(set(...)) expressions.  Express finite membership
        # using primitive comparisons instead, preserving exact semantics and
        # avoiding a parser-dependent set AST.
        comparison = "eq" if name == "in" else "ne"
        connective = "or" if name == "in" else "and"
        atom = _expr(value, names)
        predicates = [f"{comparison}({atom},{v})" for v in values]
        return predicates[0] if len(predicates) == 1 else f"{connective}({','.join(predicates)})"
    if name == "sqr" and len(args) == 1:
        value = _expr(args[0], names)
        return f"mul({value},{value})"
    if name == "wsum":
        weights, terms = args
        if len(weights) != len(terms):
            raise ValueError("Malformed weighted sum")
        parts = [f"mul({int(w)},{_expr(term, names)})" for w, term in zip(weights, terms)]
        return f"add({','.join(parts)})" if len(parts) > 1 else (parts[0] if parts else "0")
    if name in {"==", "!=", "eq", "ne"} and len(args) > 2:
        # Importer eq/ne callbacks use AllEqual / not AllEqual for n-ary forms.
        comparisons = [f"eq({_expr(args[0], names)},{_expr(arg, names)})"
                       for arg in args[1:]]
        all_equal = f"and({','.join(comparisons)})"
        return all_equal if name in {"==", "eq"} else f"not({all_equal})"
    if name == "-" and len(args) == 2:
        return f"sub({_expr(args[0], names)},{_expr(args[1], names)})"
    if name in _UNARY and len(args) == 1:
        return f"{_UNARY[name]}({_expr(args[0], names)})"
    if name in _BINARY and len(args) >= 2:
        if name in {"sub", "div", "mod", "pow", "->"} and len(args) != 2:
            raise NotImplementedError(f"{name} requires exactly two operands")
        return f"{_BINARY[name]}({','.join(_expr(a, names) for a in args)})"
    if name in _NARY and args:
        return _expr(args[0], names) if len(args) == 1 else f"{_NARY[name]}({','.join(_expr(a, names) for a in args)})"
    raise NotImplementedError(f"Unsupported XCSP3 operator: {name!r}")


def _table_constraint(parent, constraint, names):
    """Write positive, negative and short extension tables."""
    variables, rows = constraint.args
    variables = list(variables)
    if len(variables) == 0 or any(not _is_variable(v) for v in variables):
        raise NotImplementedError("Extension tables require a nonempty list of variables")

    short = constraint.name == "short_table"
    negative = constraint.name == "negative_table"
    tuples = []
    for row in rows:
        cells = list(row)
        if len(cells) != len(variables):
            raise ValueError("Table tuple length differs from variable count")
        converted = []
        for cell in cells:
            if short and isinstance(cell, str) and cell == "*":
                converted.append("*")
            elif isinstance(cell, (int, np.integer, bool, np.bool_)):
                converted.append(str(int(cell)))
            else:
                raise NotImplementedError(f"Unsupported table entry: {cell!r}")
        tuples.append("(" + ",".join(converted) + ")")

    elt = ET.SubElement(parent, "extension")
    ET.SubElement(elt, "list").text = " ".join(_expr(v, names) for v in variables)
    ET.SubElement(elt, "conflicts" if negative else "supports").text = " ".join(tuples)


def _gcc_constraint(parent, constraint, names):
    """Serialize CPMpy GlobalCardinalityCount as XCSP3 cardinality."""
    variables, values, occurrences = constraint.args
    if len(variables) == 0 or any(not _is_variable(v) for v in variables):
        raise NotImplementedError("Cardinality currently requires variable arguments")
    if len(values) != len(occurrences):
        raise ValueError("Cardinality values/occurrences length mismatch")
    if not all(isinstance(v, (int, np.integer)) for v in values):
        raise NotImplementedError("Cardinality values must be integers")

    elt = ET.SubElement(parent, "cardinality")
    if getattr(constraint, "closed", False):
        elt.set("closed", "true")
    ET.SubElement(elt, "list").text = " ".join(_expr(v, names) for v in variables)
    ET.SubElement(elt, "values").text = " ".join(str(int(v)) for v in values)
    ET.SubElement(elt, "occurs").text = " ".join(_expr(v, names) for v in occurrences)


def _ordered_constraint(parent, constraint, names):
    """Serialize ordered sequences with optional inter-element lengths."""
    operators = {
        "increasing": "le", "strictly_increasing": "lt",
        "decreasing": "ge", "strictly_decreasing": "gt",
    }
    if getattr(constraint, "lengths", None) is not None:
        values = list(constraint.args)
        lengths = list(constraint.lengths)
        if len(lengths) != len(values) - 1:
            raise ValueError("Ordered lengths must have one entry per adjacent pair")
        elt = ET.SubElement(parent, "ordered")
        _sequence(elt, "list", values, names)
        _sequence(elt, "lengths", lengths, names)
        ET.SubElement(elt, "operator").text = operators[constraint.name]
        return
    elt = ET.SubElement(parent, "ordered")
    ET.SubElement(elt, "list").text = " ".join(_expr(v, names) for v in constraint.args)
    ET.SubElement(elt, "operator").text = operators[constraint.name]


def _sequence(parent, tag, values, names):
    elt = ET.SubElement(parent, tag)
    elt.text = " ".join(_expr(v, names) for v in values)
    return elt


def _lex_constraint(parent, constraint, names):
    tag = constraint.name
    if tag in {"lex_less", "lex_lesseq"}:
        rows = list(constraint.args)
    else:
        rows = list(constraint.args)
    if len(rows) < 2:
        raise ValueError("Lex requires at least two sequences")
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Lex rows must have equal length")
    elt = ET.SubElement(parent, "lex")
    for row in rows:
        _sequence(elt, "list", row, names)
    ET.SubElement(elt, "operator").text = "lt" if tag in {"lex_less", "lex_chain_less"} else "le"


def _automaton_constraint(parent, constraint, names):
    if constraint.name == "regular":
        variables, transitions, start, accepting = constraint.args
        elt = ET.SubElement(parent, "regular")
        _sequence(elt, "list", variables, names)
        ET.SubElement(elt, "transitions").text = " ".join(
            f"({src},{int(value)},{dst})" for src, value, dst in transitions)
        ET.SubElement(elt, "start").text = str(start)
        ET.SubElement(elt, "final").text = " ".join(str(s) for s in accepting)
    else:
        variables = constraint.args[0]
        elt = ET.SubElement(parent, "mdd")
        _sequence(elt, "list", variables, names)
        transitions = [(src, value, dst)
                       for src, successors in constraint.mapping.items()
                       for value, dst in successors.items()]
        transitions.sort(key=lambda edge: edge[0] != constraint.root_node)
        ET.SubElement(elt, "transitions").text = " ".join(
            f"({src},{int(value)},{dst})" for src, value, dst in transitions)


def _scheduling_constraint(parent, constraint, names):
    args = constraint.args
    if constraint.name == "no_overlap":
        start, duration = args[:2]
        elt = ET.SubElement(parent, "noOverlap")
        _sequence(elt, "origins", start, names)
        _sequence(elt, "lengths", duration, names)
        # Explicit end variables impose additional equalities.
        if len(args) == 3:
            for s, d, end in zip(start, duration, args[2]):
                ET.SubElement(parent, "intension").text = f"eq(add({_expr(s,names)},{_expr(d,names)}),{_expr(end,names)})"
    else:
        if len(args) == 4:
            start, duration, demand, capacity = args
            end = None
        else:
            start, duration, end, demand, capacity = args
        elt = ET.SubElement(parent, "cumulative")
        _sequence(elt, "origins", start, names)
        _sequence(elt, "lengths", duration, names)
        _sequence(elt, "heights", demand, names)
        condition = getattr(constraint, "condition", None)
        if condition is None:
            operator, bound = "le", capacity
        else:
            if not isinstance(condition, (tuple, list)) or len(condition) != 2:
                raise NotImplementedError("Cumulative condition must be an (operator, bound) pair")
            operator, bound = condition
            operator = {"<": "lt", "<=": "le", "==": "eq", "!=": "ne",
                        ">=": "ge", ">": "gt"}.get(operator, operator)
            if operator not in {"lt", "le", "eq", "ne", "ge", "gt"}:
                raise NotImplementedError(f"Unsupported cumulative operator: {operator}")
        ET.SubElement(elt, "condition").text = f"({operator},{_expr(bound,names)})"
        if end is not None:
            for s, d, e in zip(start, duration, end):
                ET.SubElement(parent, "intension").text = f"eq(add({_expr(s,names)},{_expr(d,names)}),{_expr(e,names)})"



def _no_overlap_2d(parent, constraint, names):
    """Write CPMpy/XCSP3 NoOverlap2d with origins and lengths as pairs.

    The importer constructs separate end variables for each axis.  Preserve
    their defining equations instead of silently dropping them.
    """
    if len(constraint.args) != 6:
        raise ValueError("NoOverlap2d requires x/y origins, lengths and ends")
    sx, dx, ex, sy, dy, ey = (list(part) for part in constraint.args)
    count = len(sx)
    if count == 0 or any(len(part) != count for part in (dx, ex, sy, dy, ey)):
        raise ValueError("NoOverlap2d arrays must have equal nonzero lengths")
    elt = ET.SubElement(parent, "noOverlap")
    ET.SubElement(elt, "origins").text = " ".join(
        f"({_expr(x, names)},{_expr(y, names)})" for x, y in zip(sx, sy))
    ET.SubElement(elt, "lengths").text = " ".join(
        f"({_expr(x, names)},{_expr(y, names)})" for x, y in zip(dx, dy))
    for starts, lengths, ends in ((sx, dx, ex), (sy, dy, ey)):
        for start, length, end in zip(starts, lengths, ends):
            ET.SubElement(parent, "intension").text = (
                f"eq(add({_expr(start, names)},{_expr(length, names)}),{_expr(end, names)})")


def _no_overlap_nd(parent, constraint, names):
    """Serialize n-dimensional boxes as XCSP3 tuples (including mixed lengths).

    Each item in origins/lengths is a row containing one entry per dimension.
    Unlike a 2-D-only CPMpy global, this is the native XCSP3 n-D form.
    """
    if len(constraint.args) not in (2, 3):
        raise ValueError("NoOverlapND requires origins and lengths, optionally ends")
    origins, lengths = (list(map(list, part)) for part in constraint.args[:2])
    if not origins or len(origins) != len(lengths) or not origins[0]:
        raise ValueError("NoOverlapND requires nonempty matching boxes")
    dimensions = len(origins[0])
    if any(len(row) != dimensions for row in origins + lengths):
        raise ValueError("NoOverlapND requires rectangular boxes")
    elt = ET.SubElement(parent, "noOverlap")
    for tag, rows in (("origins", origins), ("lengths", lengths)):
        ET.SubElement(elt, tag).text = " ".join(
            "(" + ",".join(_expr(value, names) for value in row) + ")" for row in rows)
    if len(constraint.args) == 3:
        ends = list(map(list, constraint.args[2]))
        if len(ends) != len(origins) or any(len(row) != dimensions for row in ends):
            raise ValueError("NoOverlapND ends must match origins")
        for start_row, length_row, end_row in zip(origins, lengths, ends):
            for start, length, end in zip(start_row, length_row, end_row):
                ET.SubElement(parent, "intension").text = (
                    f"eq(add({_expr(start, names)},{_expr(length, names)}),{_expr(end, names)})")


def _circuit_constraint(parent, constraint, names):
    """XCSP3 circuit uses zero-based successor indices by default."""
    successors = list(constraint.args)
    if len(successors) < 2 or any(not _is_variable(v) for v in successors):
        raise NotImplementedError("Circuit requires at least two successor variables")
    elt = ET.SubElement(parent, "circuit")
    _sequence(elt, "list", successors, names)
    # The importer uses SubCircuitWithStart, which otherwise permits
    # self-loops; CPMpy Circuit requires a Hamiltonian cycle.
    for index, successor in enumerate(successors):
        ET.SubElement(parent, "intension").text = (
            f"ne({_expr(successor, names)},{index})")


def _all_different_except(parent, sequence, except_values, names):
    elt = ET.SubElement(parent, "allDifferent")
    _sequence(elt, "list", sequence, names)
    if except_values:
        ET.SubElement(elt, "except").text = " ".join(str(int(v)) for v in except_values)


def _sum_condition(parent, terms, operator, rhs, names, coefficients=None):
    elt = ET.SubElement(parent, "sum")
    _sequence(elt, "list", terms, names)
    if coefficients is not None:
        _sequence(elt, "coeffs", coefficients, names)
    ET.SubElement(elt, "condition").text = f"({operator},{_expr(rhs,names)})"


def _decomposed_constraint(parent, constraint, names):
    """Export a decomposition, including its auxiliary defining constraints.

    Only a writer registry can declare new variables. Standalone white-box
    calls with a plain name map keep rejecting undeclared auxiliaries.
    """
    try:
        decomposed, defining = constraint.decompose()
    except (AttributeError, NotImplementedError) as exc:
        raise NotImplementedError(f"Cannot export {constraint.name}") from exc
    if defining and not isinstance(names, _VariableRegistry):
        raise NotImplementedError(f"{constraint.name} needs auxiliary variables")
    if isinstance(names, _VariableRegistry):
        names.register(get_variables(list(decomposed) + list(defining)))
    for c in list(defining) + list(decomposed):
        _constraint(parent, c, names)


def _weighted_load(assignments, sizes, bin_number, names):
    """Load of one bin, represented as an XCSP3 intension expression."""
    if len(assignments) != len(sizes):
        raise ValueError("BinPacking assignments and sizes must have equal lengths")
    terms = []
    for assignment, size in zip(assignments, sizes):
        if not isinstance(size, (int, np.integer)):
            raise NotImplementedError("BinPacking sizes must be integer constants")
        terms.append(f"mul({int(size)},eq({_expr(assignment, names)},{bin_number}))")
    return _add(terms)


def _add(terms):
    return "add(" + ",".join(terms) + ")" if len(terms) > 1 else (terms[0] if terms else "0")


def _condition_text(condition, names):
    """Condition tuple (operator, right hand side), without silently guessing."""
    if not isinstance(condition, (tuple, list)) or len(condition) != 2:
        raise NotImplementedError("Condition must be (operator, right-hand side)")
    operator, rhs = condition
    aliases = {"==": "eq", "!=": "ne", "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}
    op = aliases.get(str(operator), str(operator))
    if op not in {"eq", "ne", "lt", "le", "gt", "ge"}:
        raise NotImplementedError(f"Unsupported comparison: {operator}")
    return op, _expr(rhs, names)


def _binpacking_constraint(parent, constraint, names):
    """Emit XCSP3 binPacking, with either per-bin limits, loads or a condition."""
    name = constraint.name
    args = constraint.args
    if len(args) != 3:
        raise ValueError("BinPacking requires assignments, sizes and limits/loads/condition")
    assignments, sizes, specification = args
    assignments, sizes = list(assignments), list(sizes)
    if not assignments or len(assignments) != len(sizes):
        raise ValueError("BinPacking requires equally sized nonempty arrays")
    if not all(_is_variable(v) for v in assignments):
        raise NotImplementedError("BinPacking assignments must be variables")
    if not all(isinstance(w, (int, np.integer)) and int(w) >= 0 for w in sizes):
        raise NotImplementedError("BinPacking sizes must be nonnegative integers")
    elt = ET.SubElement(parent, "binPacking")
    _sequence(elt, "list", assignments, names)
    ET.SubElement(elt, "sizes").text = " ".join(str(int(w)) for w in sizes)
    if name in {"binpacking_limits", "binpacking_loads"}:
        values = list(specification)
        if not values:
            raise ValueError("BinPacking limits/loads must be nonempty")
        _sequence(elt, "limits" if name.endswith("limits") else "loads", values, names)
    else:
        op, rhs = _condition_text(specification, names)
        ET.SubElement(elt, "condition").text = f"({op},{rhs})"


def _knapsack_constraint(parent, constraint, names):
    """Export binary selection with two XCSP3 conditions."""
    if len(constraint.args) != 5:
        raise ValueError("Knapsack requires variables, weights, weight condition, profits, profit condition")
    variables, weights, weight_condition, profits, profit_condition = constraint.args
    variables, weights, profits = list(variables), list(weights), list(profits)
    if not variables or len(variables) != len(weights) or len(variables) != len(profits):
        raise ValueError("Knapsack arrays must be equally sized and nonempty")
    if any(not _is_variable(v) for v in variables):
        raise NotImplementedError("Knapsack requires variable selections")
    if any(not isinstance(w, (int, np.integer)) for w in weights + profits):
        raise NotImplementedError("Knapsack weights and profits must be integer constants")
    wop, wrhs = _condition_text(weight_condition, names)
    pop, prhs = _condition_text(profit_condition, names)
    elt = ET.SubElement(parent, "knapsack")
    _sequence(elt, "list", variables, names)
    ET.SubElement(elt, "weights").text = " ".join(map(str, weights))
    ET.SubElement(elt, "condition").text = f"({wop},{wrhs})"
    ET.SubElement(elt, "profits").text = " ".join(map(str, profits))
    ET.SubElement(elt, "condition").text = f"({pop},{prhs})"


def _channel_constraint(parent, constraint, names):
    args = constraint.args
    elt = ET.SubElement(parent, "channel")
    if constraint.name in {"inverse_one", "inverseone"}:
        _sequence(elt, "list", args[0] if len(args) == 1 else args, names)
    elif constraint.name in {"channel_value", "channel"}:
        if len(args) != 2:
            raise ValueError("Channel requires a list and a value")
        _sequence(elt, "list", args[0], names)
        ET.SubElement(elt, "value").text = _expr(args[1], names)
    else:
        if len(args) != 2:
            raise ValueError("Inverse requires two lists")
        _sequence(elt, "list", args[0], names)
        _sequence(elt, "list", args[1], names)


def _all_different_lists(parent, constraint, names):
    """Tuple-wise allDifferent, optionally with excluded tuples."""
    if constraint.name in {"alldifferent_lists_except_n", "all_different_lists_except_n"}:
        rows, excluded = constraint.args
    else:
        rows, excluded = constraint.args, None
    rows = list(rows)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("AllDifferentLists requires equally sized nonempty lists")
    elt = ET.SubElement(parent, "allDifferent")
    for row in rows:
        _sequence(elt, "list", row, names)
    if excluded is not None:
        ET.SubElement(elt, "except").text = " ".join(
            "(" + ",".join(str(int(v)) for v in row) + ")" for row in excluded)


def _matrix_constraint(parent, constraint, names):
    """Represent row-and-column global constraints using explicit XCSP3 matrix."""
    if constraint.name in {"alldifferent_matrix", "all_different_matrix"}:
        matrix = constraint.args[0]
        tag = "allDifferent"
        operator = None
    else:
        matrix, operator = constraint.args
        tag = "lex"
    matrix = list(matrix)
    if not matrix or not matrix[0] or any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError("Matrix must be nonempty and rectangular")
    elt = ET.SubElement(parent, tag)
    matrix_node = ET.SubElement(elt, "matrix")
    matrix_node.text = " ".join("(" + ",".join(_expr(v, names) for v in row) + ")" for row in matrix)
    if operator is not None:
        normalized = {"<": "lt", "<=": "le", "lt": "lt", "le": "le"}.get(str(operator))
        if normalized is None:
            raise NotImplementedError(f"Unsupported lex matrix operator: {operator}")
        ET.SubElement(elt, "operator").text = normalized


def _element_matrix_constraint(parent, constraint, names):
    """Serialize XCSP3 matrix element with two variable indices."""
    matrix, row_index, col_index, operator, rhs = constraint.args
    matrix = [list(row) for row in matrix]
    if not matrix or not matrix[0] or any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError("ElementMatrix requires a nonempty rectangular matrix")
    if (not _is_variable(row_index) or not _is_variable(col_index)
            or row_index.lb < 0 or row_index.ub >= len(matrix)
            or col_index.lb < 0 or col_index.ub >= len(matrix[0])):
        raise NotImplementedError("ElementMatrix requires in-range variable indices")
    op = _BINARY.get(operator)
    if op not in {"eq", "ne", "lt", "le", "gt", "ge"}:
        raise NotImplementedError(f"Unsupported ElementMatrix operator: {operator}")
    # The existing XCSP3 importer does not preserve the semantics of
    # rectangular matrix elements. Use an equivalent intension expression
    # until its native rectangular-matrix handling is corrected.
    if len(matrix) != len(matrix[0]):
        row_xml = _expr(row_index, names)
        col_xml = _expr(col_index, names)
        rhs_xml = _expr(rhs, names)
        cases = [
            f"and(eq({row_xml},{r}),eq({col_xml},{c}),"
            f"{op}({_expr(value, names)},{rhs_xml}))"
            for r, values in enumerate(matrix)
            for c, value in enumerate(values)
        ]
        ET.SubElement(parent, "intension").text = (
            cases[0] if len(cases) == 1 else f"or({','.join(cases)})"
        )
        return
    elt = ET.SubElement(parent, "element")
    ET.SubElement(elt, "matrix").text = " ".join(
        "(" + ",".join(_expr(v, names) for v in row) + ")" for row in matrix
    )
    ET.SubElement(elt, "index").text = f"{_expr(row_index, names)} {_expr(col_index, names)}"
    ET.SubElement(elt, "condition").text = f"({op},{_expr(rhs, names)})"


def _element_comparison(parent, constraint, names):
    """Use XCSP3 element for a top-level comparison, avoiding nested numeric ifs."""
    op = getattr(constraint, "name", None)
    if op not in {"==", "!=", "<", "<=", ">", ">="}:
        return False
    args = getattr(constraint, "args", ())
    if len(args) != 2:
        return False
    left, right = args
    element = left if getattr(left, "name", None) in {"element", "nd_element"} else right if getattr(right, "name", None) in {"element", "nd_element"} else None
    if element is None:
        return False
    if element is right:
        op = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}.get(op, op)
        rhs = left
    else:
        rhs = right
    if element.name == "nd_element":
        matrix, *indices = element.args
        if len(indices) != 2 or getattr(matrix, "ndim", None) != 2:
            return False
        row_index, col_index = indices
        if (not _is_variable(row_index) or not _is_variable(col_index)
                or row_index.lb < 0 or row_index.ub >= matrix.shape[0]
                or col_index.lb < 0 or col_index.ub >= matrix.shape[1]):
            return False
        _element_matrix_constraint(parent,
                                   type("MatrixElement", (), {"args": (matrix, row_index, col_index, op, rhs)})(),
                                   names)
        return True
    array, index = element.args
    array = list(array)
    if not array or not _is_variable(index) or index.lb < 0 or index.ub >= len(array):
        return False
    elt = ET.SubElement(parent, "element")
    _sequence(elt, "list", array, names)
    ET.SubElement(elt, "index").text = _expr(index, names)
    ET.SubElement(elt, "condition").text = f"({_BINARY[op]},{_expr(rhs, names)})"
    return True


def _constraint(parent, constraint, names):
    name = getattr(constraint, "name", None)
    # The XCSP3 importer represents a negative extension as ~cp.Table(...).
    # Serialize that form as a native conflicts table rather than trying to
    # encode the table as an intension expression.
    if name in {"not", "~"} and len(getattr(constraint, "args", ())) == 1:
        inner = constraint.args[0]
        if getattr(inner, "name", None) == "table":
            class _NegativeTableView:
                name = "negative_table"
                args = inner.args
            _table_constraint(parent, _NegativeTableView(), names)
            return
    if _element_comparison(parent, constraint, names):
        return
    if name in {"table", "short_table", "negative_table"}:
        _table_constraint(parent, constraint, names)
    elif name == "element_matrix":
        _element_matrix_constraint(parent, constraint, names)
    elif name == "instantiation":
        if len(constraint.args) != 2:
            raise ValueError("Instantiation requires a list of variables and a list of values")
        variables, values = map(list, constraint.args)
        if not variables or len(variables) != len(values):
            raise ValueError("Instantiation requires nonempty lists of equal length")
        if not all(_is_variable(var) for var in variables):
            raise NotImplementedError("Instantiation requires variables")
        if not all(isinstance(value, (int, np.integer)) for value in values):
            raise NotImplementedError("Instantiation requires integer values")
        elt = ET.SubElement(parent, "instantiation")
        _sequence(elt, "list", variables, names)
        ET.SubElement(elt, "values").text = " ".join(str(int(value)) for value in values)
    elif name == "circuit":
        _circuit_constraint(parent, constraint, names)
    elif name == "gcc":
        _gcc_constraint(parent, constraint, names)
    elif name in {"increasing", "strictly_increasing", "decreasing", "strictly_decreasing"}:
        _ordered_constraint(parent, constraint, names)
    elif name in {"lex_less", "lex_lesseq", "lex_chain_less", "lex_chain_lesseq"}:
        _lex_constraint(parent, constraint, names)
    elif name in {"regular", "mdd"}:
        _automaton_constraint(parent, constraint, names)
    elif name in {"no_overlap", "cumulative"}:
        _scheduling_constraint(parent, constraint, names)
    elif name in {"no_overlap_2d", "nooverlap2d"}:
        _no_overlap_2d(parent, constraint, names)
    elif name in {"no_overlap_nd", "nooverlap_nd", "no_overlap_multi"}:
        _no_overlap_nd(parent, constraint, names)
    elif name in {"inverse", "inverse_one", "inverseone", "safe_only_inverse", "channel", "channel_value"}:
        _channel_constraint(parent, constraint, names)
    elif name in {"binpacking", "binpacking_limits", "binpacking_loads"}:
        _binpacking_constraint(parent, constraint, names)
    elif name == "knapsack":
        _knapsack_constraint(parent, constraint, names)
    elif name in {"alldifferent_lists", "all_different_lists", "alldifferent_lists_except_n", "all_different_lists_except_n"}:
        _all_different_lists(parent, constraint, names)
    elif name in {"alldifferent_matrix", "all_different_matrix", "lex_matrix"}:
        _matrix_constraint(parent, constraint, names)
    elif name == "precedence":
        variables, values = constraint.args
        elt = ET.SubElement(parent, "precedence")
        _sequence(elt, "list", variables, names)
        ET.SubElement(elt, "values").text = " ".join(str(int(v)) for v in values)
    elif name in _GLOBAL:
        if len(constraint.args) == 0:
            raise ValueError(f"Empty global constraint: {name}")
        elt = ET.SubElement(parent, _GLOBAL[name])
        ET.SubElement(elt, "list").text = " ".join(_expr(arg, names) for arg in constraint.args)
    elif name in {"all_different_except", "alldifferent_except", "alldifferent_except_n", "alldifferent_except_0"}:
        if name == "alldifferent_except_0":
            variables, except_values = constraint.args[0], [0]
        else:
            variables, except_values = constraint.args
        _all_different_except(parent, variables, except_values, names)
    elif name == "allequal_except_n":
        _decomposed_constraint(parent, constraint, names)
    elif name in {"indomain", "ite", "xor"}:
        ET.SubElement(parent, "intension").text = _expr(constraint, names)
    else:
        try:
            serialized = _expr(constraint, names)
        except NotImplementedError:
            # Prefer a CPMpy-provided decomposition when the construct is
            # not representable directly. Unknown constructs still fail closed.
            if hasattr(constraint, "decompose"):
                _decomposed_constraint(parent, constraint, names)
            else:
                raise
        else:
            ET.SubElement(parent, "intension").text = serialized


def _explicit_variable_domains(constraints):
    """Extract domain holes from top-level positive InDomain constraints.

    Keep the constraints as well: narrowing the XML declaration is a redundant
    representation, not a replacement for a constraint.  Never infer holes
    from arbitrary Boolean subexpressions or negated membership.
    """
    domains = {}
    for constraint in constraints:
        if getattr(constraint, "name", None) != "indomain":
            continue
        args = getattr(constraint, "args", ())
        if len(args) != 2 or not _is_variable(args[0]):
            continue
        variable, allowed = args
        if not all(isinstance(v, (int, np.integer)) for v in allowed):
            raise NotImplementedError("InDomain requires integer domain values")
        valid = {int(v) for v in allowed if variable.lb <= int(v) <= variable.ub}
        key = id(variable)
        domains[key] = valid if key not in domains else domains[key] & valid
    return domains


def _domain_text(variable, explicit):
    if explicit is None:
        return f"{int(variable.lb)}..{int(variable.ub)}"
    if not explicit:
        # Empty domains are not representable as an XCSP3 variable declaration.
        # Preserve the original interval and let the InDomain intension
        # express the contradiction instead.
        return f"{int(variable.lb)}..{int(variable.ub)}"
    values = sorted(explicit)
    if values == list(range(values[0], values[-1] + 1)):
        return f"{values[0]}..{values[-1]}"
    return " ".join(map(str, values))


def _objective_xml(parent, objective, names):
    """Use XCSP3 specialized objectives for directly matching CPMpy forms."""
    name = getattr(objective, "name", None)
    args = getattr(objective, "args", ())
    if name in {"sum", "min", "max", "nvalue"} and args:
        types = {"sum": "sum", "min": "minimum", "max": "maximum", "nvalue": "nValues"}
        parent.set("type", types[name])
        _sequence(parent, "list", args, names)
        # The supported pycsp3 parser expects a second child for every
        # specialized objective, including maximum, minimum and nValues.
        # Unit coefficients leave these three objective values unchanged.
        _sequence(parent, "coeffs", [1] * len(args), names)
    elif name == "wsum" and len(args) == 2:
        weights, terms = args
        if len(weights) != len(terms):
            raise ValueError("Malformed weighted objective")
        parent.set("type", "sum")
        _sequence(parent, "list", terms, names)
        _sequence(parent, "coeffs", weights, names)
    else:
        parent.text = _expr(objective, names)


def write_xcsp3(model, destination):
    """Export supported CPMpy models to an XCSP3 XML file.

    Unsupported constructs raise NotImplementedError before the file is written.
    CPMpy array-element names are mapped to valid independent XCSP3 IDs.
    """
    constraints = list(model.constraints)
    objective = model.objective_ if model.has_objective() else None
    expressions = constraints + ([objective] if objective is not None else [])
    variables = list(get_variables(expressions))
    names = _VariableRegistry(variables)
    explicit_domains = _explicit_variable_domains(constraints)

    root = ET.Element("instance", {"format": "XCSP3", "type": "COP" if objective is not None else "CSP"})
    cons_xml = ET.SubElement(root, "constraints")
    for constraint in constraints:
        _constraint(cons_xml, constraint, names)

    if objective is not None:
        objectives_xml = ET.SubElement(root, "objectives")
        tag = "minimize" if model.objective_is_min else "maximize"
        _objective_xml(ET.SubElement(objectives_xml, tag), objective, names)

    # The constraints may have introduced auxiliary variables. Declare all of
    # them before the constraints in the final XCSP3 document.
    var_xml = ET.Element("variables")
    for var in names.variables:
        ET.SubElement(var_xml, "var", {"id": names[id(var)]}).text = _domain_text(
            var, explicit_domains.get(id(var)))
    root.insert(0, var_xml)

    ET.indent(root, space="  ")
    output = '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode") + '\n'
    Path(destination).write_text(output, encoding="utf-8")
