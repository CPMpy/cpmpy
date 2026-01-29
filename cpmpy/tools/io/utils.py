import warnings


# mapping file extensions to appropriate format names
_format_map = {
    "mps"   : "mps",
    "lp"    : "lp",
    "cip"   : "cip",
    "fzn"   : "fzn",
    "gms"   : "gms",
    "pip"   : "pip",
    "wcnf"  : "dimacs",
    "cnf"   : "dimacs",
}

_extension_map = {}
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