"""
Dataset utilities.
"""

import json
import pathlib
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Union
from urllib.request import Request, urlopen


from .metadata import (
    InstanceInfo, DatasetInfo, FeaturesInfo, FieldInfo,
    _MODEL_FEATURE_FIELDS, _FORMAT_SPECIFIC_PREFIXES,
)


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
        and k not in _MODEL_FEATURE_FIELDS
        and not any(k.startswith(p) for p in _FORMAT_SPECIFIC_PREFIXES)
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


def _init_metadata_worker(context_dict, collect_metadata_func, reader_func, open_func):
    """Initialize worker process with dataset context."""
    global _metadata_worker_context
    _metadata_worker_context = context_dict.copy()
    _metadata_worker_context['collect_instance_metadata'] = collect_metadata_func
    _metadata_worker_context['reader'] = reader_func
    _metadata_worker_context['open_func'] = open_func


def _collect_one_metadata_worker(file_path_str):
    """Worker function for process-based metadata collection."""
    global _metadata_worker_context
    file_path = pathlib.Path(file_path_str)
    dataset_dir = pathlib.Path(_metadata_worker_context['dataset_dir'])
    meta_path = dataset_dir / (file_path.name + _metadata_worker_context['metadata_extension'])
    
    # Collect instance metadata using the provided function
    collect_metadata = _metadata_worker_context['collect_instance_metadata']
    metadata_error = None
    try:
        instance_meta = collect_metadata(str(file_path))
    except Exception as e:
        instance_meta = {}
        metadata_error = str(e)

    # Separate portable from format-specific fields
    portable = portable_instance_metadata(instance_meta)
    format_specific = {
        k: v for k, v in instance_meta.items()
        if k not in portable and not k.startswith("_")
    }

    # Derive instance name
    stem = file_path.stem
    for ext in (".xml", ".wcnf", ".opb"):
        if stem.endswith(ext):
            stem = stem[:len(stem) - len(ext)]
            break

    # Build structured sidecar
    sidecar = {
        "dataset": _metadata_worker_context['dataset_metadata'],
        "instance_name": stem,
        "source_file": str(file_path.relative_to(dataset_dir)),
        "category": _metadata_worker_context['category'],
        "instance_metadata": portable,
        "format_metadata": format_specific,
    }

    if metadata_error is not None:
        sidecar["_metadata_error"] = metadata_error

    # Preserve or compute model features
    model_features = None
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                existing = json.load(f)
            if "model_features" in existing:
                model_features = existing["model_features"]
        except (json.JSONDecodeError, IOError):
            pass

    if model_features is None:
        reader = _metadata_worker_context['reader']
        open_func = _metadata_worker_context['open_func']
        if not callable(reader):
            raise TypeError(
                f"Cannot extract model features for {file_path}: "
                "no dataset reader configured."
            )
        model = reader(str(file_path), open=open_func)
        model_features = extract_model_features(model)
    
    sidecar["model_features"] = model_features

    with open(meta_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    
    return str(file_path)


# ---------------------------------------------------------------------------- #
#                              Download utilities.                             #
# ---------------------------------------------------------------------------- #

def _get_content_length(url: str) -> int:
    """
    Return Content-Length for url, or 0 if unknown.
    """
    try:
        req = Request(url)
        req.get_method = lambda: "HEAD"
        with urlopen(req) as resp:
            return int(resp.headers.get("Content-Length", 0))
    except Exception:
        return 0

def _download_url(
    url: str,
    destination: Union[str, pathlib.Path],
    desc: str = None,
    chunk_size: int = 1024 * 1024,
    _sequential_impl=None,
) -> pathlib.Path:
    """
    Download a single file from url to destination.
    Uses _sequential_impl(url, path, total_size, desc, chunk_size) if provided,
    otherwise delegates to the dataset base implementation.
    """
    destination = pathlib.Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if desc is None:
        desc = destination.name
    total_size = _get_content_length(url)
    if _sequential_impl is None:
        from cpmpy.tools.datasets.core import FileDataset
        _sequential_impl = FileDataset._download_sequential
    _sequential_impl(url, destination, total_size, desc, chunk_size)
    return destination


def download_manager(
    url: Union[str, List[str]],
    destination: Union[str, pathlib.Path, List[Union[str, pathlib.Path]]] = None,
    *,
    workers: int = 1,
    desc_prefix: str = "Downloading",
    chunk_size: int = 1024 * 1024,
    skip_existing: bool = True,
    **kwargs,
) -> Union[pathlib.Path, List[pathlib.Path]]:
    """
    Generic download manager: one URL or many, sequential or parallel.

    Single file:
        path = download("https://example.com/file.zip", "/tmp/file.zip")
        path = download("https://example.com/file.zip", destination="/tmp/out.zip", workers=1)

    Multiple files (list of (url, destination)):
        paths = download([("https://a.com/1.cnf", "/data/1.cnf"), ...], workers=4)

    Arguments:
        url: Either a single URL string, or a list of URL strings.
        destination: For single-URL mode, path to save the file. For multiple-URL mode, list of matching destination paths.
        workers: Number of parallel download workers. 1 = sequential. >1 = parallel (only for multiple files).
        desc_prefix: Prefix for progress description (e.g. "Instance 1/100").
        chunk_size: Chunk size in bytes for streaming.
        skip_existing: If True, skip pairs where destination already exists (multi-file only).
        **kwargs: Ignored; allows callers to pass through options (e.g. from dataset download(**kwargs)).

    Returns:
        For single URL: path to the downloaded file.
        For multiple: list of paths that were downloaded (skipped files are not in the list).
    """
    if isinstance(url, str):
        if destination is None:
            raise ValueError("destination is required when passing a single URL")
        return _download_url(url, destination, desc=desc_prefix or url, chunk_size=chunk_size)

    items: List[Tuple[str, pathlib.Path]] = [
        (url, pathlib.Path(dest)) for url, dest in zip(url, destination)
    ]

    if not items:
        return []

    if skip_existing:
        items = [(u, d) for u, d in items if not d.exists()]

    if not items:
        return []

    if workers is None or workers <= 1:
        # Sequential
        results = []
        for i, (url, dest) in enumerate(items):
            desc = f"{desc_prefix} {i + 1}/{len(items)} {dest.name}"
            try:
                results.append(_download_url(url, dest, desc=desc, chunk_size=chunk_size))
            except Exception as e:
                warnings.warn(f"Failed to download {url}: {e}")
        return results

    # Parallel
    max_workers = min(workers, len(items))
    results = []
    errors = []

    def do_one(url: str, dest: pathlib.Path, idx: int) -> Tuple:
        desc = f"{desc_prefix} {idx + 1}/{len(items)} {dest.name}"
        try:
            return _download_url(url, dest, desc=desc, chunk_size=chunk_size), None
        except Exception as e:
            return None, str(e)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(do_one, url, dest, i): (url, dest)
            for i, (url, dest) in enumerate(items)
        }
        for future in as_completed(futures):
            result, err = future.result()
            if result is not None:
                results.append(result)
            else:
                url, dest = futures[future]
                errors.append((dest.name, err))

    if errors:
        warnings.warn(
            f"Failed to download {len(errors)}/{len(items)} files. "
            f"First error: {errors[0][0]} - {errors[0][1]}"
        )

    return results


# Convenience alias for multi-file callers
download_many = download_manager
