import warnings
import os
from typing import Union

# mapping file extensions to appropriate format names
_format_map = {
    "mps"   : "mps",
    "lp"    : "lp",
    "cip"   : "cip",
    "fzn"   : "fzn",
    "gms"   : "gms",
    "pip"   : "pip",
    "wcnf"  : "wcnf",
    "cnf"   : "cnf",
    "opb"   : "opb",
}

_extension_map: dict[str, list[str]] = {}
for extension, format in _format_map.items():
    _extension_map[format] = _extension_map.get(format, []) + [extension]

def get_extension(format: str) -> str:
    """
    Get the file extension for a given format.
    """
    if len(_extension_map[format]) > 1:
        warnings.warn(f"Multiple extensions found for format {format}: {_extension_map[format]}. Using the first one: {_extension_map[format][0]}")
    
    return _extension_map[format][0]

def get_format(extension: str) -> str:
    """
    Get the format for a given file extension.
    """
    return _format_map[extension]

def _derive_format(file_path: os.PathLike) -> str:
    """
    Derive the format of a file from its path by looking at its file extension.

    Arguments:
        file_path (str): The path to the file to derive the format from.

    Raises:
        ValueError: If the format could not be derived from the file path.

    Returns:
        The name of the format.

    Example:
        >>> _derive_format("instance.mps")
        "mps"
        >>> _derive_format("instance.lp.xz")
        "lp"
        >>> _derive_format("instance.cnf")
        "dimacs"
    """

    # Iterate over the file path extensions in reverse order
    for ext in str(file_path).split(".")[::-1]:
        try:
            return get_format(ext)
        except ValueError:
            continue

    raise ValueError(f"No file format provided and could not derive format from file path: {file_path}")

def _is_potential_path(instance: Union[str, os.PathLike]) -> bool:
    """
    Check if a given instance is a potential path.
    """

    is_pathlike = isinstance(instance, os.PathLike)
    is_string = isinstance(instance, str)

    if is_pathlike:
        return True

    if not is_pathlike and not is_string:
        raise ValueError("Instance must be a string or a path-like object")

    if is_string:
        if ("\n" in instance or "\r" in instance): # typical indicator of inline contents, not present in a path string
            return False
        return True

    return False