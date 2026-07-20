"""
Input/Output (IO) tools for CPMpy.

This module provides tools to load and write models in various formats.
Use the generic ``load(..., format="...")`` and ``write(..., format="...")`` functions to load and write
models in one of the supported formats.

Some formats can be auto-detected from the file extension, so only a file path is required as argument.

Loading models
---------------

Use :func:`load` to read a model from a file path, an open text file, or a string containing the
instance data. When loading from a file path, the format can often be inferred from the extension.
When loading from a string, pass ``format=`` explicitly.

.. code-block:: python

    from cpmpy.tools.io import load

    model = load("instance.opb")                         # format inferred from ".opb"
    model = load("instance.txt", format="opb")           # explicit format
    model = load("* #variable= 1 #constraint= 1\\n+1 x1 >= 1;", format="opb")

Format-specific loaders are also available when you want to call them directly:

.. code-block:: python

    from cpmpy.tools.io import load_opb, load_dimacs

    model = load_opb("instance.opb")
    cnf_model = load_dimacs("p cnf 2 1\\n1 -2 0\\n")

More on loading formats: :doc:`loader and writer API documentation </api/tools/io/loader>`.

Writing models
--------------

Use :func:`write` to export a CPMpy model. The serialized model is returned as a string. 
If ``path`` is provided, the same string also gets written to disk.

.. code-block:: python

    import cpmpy as cp
    from cpmpy.tools.io import write

    x = cp.boolvar(shape=3, name="x")
    model = cp.Model(cp.sum(x) >= 2)

    text = write(model, format="opb")                    # return a string
    text = write(model, "model.opb")                     # infer format from extension
    text = write(model, "model.txt", format="opb")       # explicit format

Format-specific writers expose format-specific options:

.. code-block:: python

    from cpmpy.tools.io import write_dimacs, write_opb

    dimacs_text = write_dimacs(model, p_header=True)
    opb_text = write_opb(model)

More on writing formats: :doc:`loader and writer API documentation </api/tools/io/writer>`.

Listing and extensions
----------------------

Use :func:`load_formats` and :func:`write_formats` to inspect the currently registered formats.

.. code-block:: python

    from cpmpy.tools.io import load_formats, write_formats

    print(load_formats()) 
    >> ['mps', 'lp', 'cip', 'fzn', 'gms', 'pip', 'dimacs', 'cnf', 'wcnf', 'opb', 'jsplib', 'rcpsp', 'nurserostering']
    print(write_formats())
    >> ['mps', 'lp', 'cip', 'fzn', 'gms', 'pip', 'dimacs', 'cnf', 'wcnf', 'opb']


Custom file opening
-------------------

The generic readers and writers accept an ``open=`` argument for custom file handling, for example
compressed files.

.. code-block:: python

    import lzma
    from cpmpy.tools.io import load, write

    model = load("instance.opb.xz", format="opb", open=lambda p, mode="r": lzma.open(p, "rt"))
    write(model, "model.opb.xz", format="opb", open=lambda p, mode="w": lzma.open(p, "wt"))
"""

# Cross-format loaders and writers + utility functions
from .writer import write, write_formats
from .loader import load, load_formats
from .utils import get_extension, get_format

# Problem-specific loaders
from .jsplib import load_jsplib        
from .nurserostering import load_nurserostering
from .rcpsp import load_rcpsp

# Standard format loaders and writers
from .opb import load_opb, write_opb
from .scip_formats import load_scip_format, write_scip_format
from .wcnf import load_wcnf
from .dimacs import load_dimacs, write_dimacs
from .gdimacs import load_gdimacs, write_gdimacs
from .xcsp3 import load_xcsp3

_all__ = [
    "load",
    "load_formats",
    "write",
    "write_formats",
    "load_opb",
    "write_opb",
    "load_scip_format",
    "write_scip_format",
    "load_dimacs",
    "write_dimacs",
    "load_gdimacs",
    "write_gdimacs",
    "load_wcnf",
    "load_xcsp3",
    "load_jsplib",
    "load_rcpsp",
    "load_nurserostering",
    "get_extension",
    "get_format",
]