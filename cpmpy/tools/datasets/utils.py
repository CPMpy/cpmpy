"""
Dataset utilities.
"""

import json
import pathlib
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Union
from urllib.request import Request, urlopen


def portable_instance_metadata(metadata: dict) -> dict:
    """
    Filter metadata to only portable, domain-specific fields.

    Strips model features (num_variables, constraint_types, ...) and
    format-specific fields (opb_*, wcnf_*, mps_*, ...) linked to a specific
    file format.

    Keeps domain-specific metadata that is independent of the file format,
    such as ``jobs``, ``machines``, ``optimum``, ``horizon``, ``bounds``, etc.

    Arguments:
        metadata (dict): Full sidecar metadata dictionary.

    Returns:
        dict with only portable fields.
    """
    return {
        k: v for k, v in metadata.items()
        if not k.startswith("_")
        # and k not in _MODEL_FEATURE_FIELDS TODO
        # and not any(k.startswith(p) for p in _FORMAT_SPECIFIC_PREFIXES)
    }

def extract_model_features(model) -> dict:
    """
    Extract generic CP features from a CPMpy Model.

    Arguments:
        model: a cpmpy.Model instance

    Returns:
        dict with keys: num_variables, num_bool_variables, num_int_variables,
        num_constraints, constraint_types, has_objective, objective_type,
        domain_size_min, domain_size_max, domain_size_mean
    """
    from cpmpy.transformations.get_variables import get_variables_model
    from cpmpy.expressions.variables import _BoolVarImpl
    from cpmpy.expressions.core import Expression
    from cpmpy.expressions.utils import is_any_list

    variables = get_variables_model(model)

    num_bool = sum(1 for v in variables if isinstance(v, _BoolVarImpl))
    num_int = len(variables) - num_bool

    # Domain sizes (lb/ub available on all variable types)
    domain_sizes = [int(v.ub) - int(v.lb) + 1 for v in variables] if variables else []

    # Constraint types: collect .name from top-level constraints
    constraint_type_counts = {}

    def _count_constraints(c):
        if is_any_list(c):
            for sub in c:
                _count_constraints(sub)
        elif isinstance(c, Expression):
            name = c.name
            constraint_type_counts[name] = constraint_type_counts.get(name, 0) + 1

    for c in model.constraints:
        _count_constraints(c)

    num_constraints = sum(constraint_type_counts.values())

    # Objective
    has_obj = model.objective_ is not None
    obj_type = "none"
    if has_obj:
        obj_type = "min" if model.objective_is_min else "max"

    return {
        "num_variables": len(variables),
        "num_bool_variables": num_bool,
        "num_int_variables": num_int,
        "num_constraints": num_constraints,
        "constraint_types": constraint_type_counts,
        "has_objective": has_obj,
        "objective_type": obj_type,
        "domain_size_min": min(domain_sizes) if domain_sizes else None,
        "domain_size_max": max(domain_sizes) if domain_sizes else None,
        "domain_size_mean": round(sum(domain_sizes) / len(domain_sizes), 2) if domain_sizes else None,
    }
