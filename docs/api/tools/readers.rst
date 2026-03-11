Loaders (:mod:`cpmpy.tools.io`)
===============================

CPMpy provides loaders for various constraint programming and optimization file
formats. All loaders accept either a file path or a raw content string, and
return a :class:`cpmpy.Model` ready to solve.

Basic Usage
-----------

A unified ``load()`` function auto-detects the format from the file extension:

.. code-block:: python

    from cpmpy.tools.io import load

    model = load("instance.opb")
    model = load("instance.cnf")
    model = load("problem.mps")

    # Explicit format when the extension is ambiguous
    model = load("instance.txt", format="opb")

    model.solve()

Supported Formats
-----------------

.. list-table::
   :header-rows: 1

   * - **Format**
     - **Extension**
     - **Loader function**
     - **Dependencies**
   * - OPB
     - ``.opb``
     - :func:`load_opb <cpmpy.tools.io.opb.load_opb>`
     - —
   * - WCNF
     - ``.wcnf``
     - :func:`load_wcnf <cpmpy.tools.io.wcnf.load_wcnf>`
     - —
   * - DIMACS
     - ``.cnf``
     - :func:`load_dimacs <cpmpy.tools.io.dimacs.load_dimacs>`
     - —
   * - MPS
     - ``.mps``
     - :func:`load_scip <cpmpy.tools.io.scip.load_scip>`
     - pyscipopt
   * - LP
     - ``.lp``
     - :func:`load_scip <cpmpy.tools.io.scip.load_scip>`
     - pyscipopt
   * - CIP
     - ``.cip``
     - :func:`load_scip <cpmpy.tools.io.scip.load_scip>`
     - pyscipopt
   * - FZN
     - ``.fzn``
     - :func:`load_scip <cpmpy.tools.io.scip.load_scip>`
     - pyscipopt
   * - GMS
     - ``.gms``
     - :func:`load_scip <cpmpy.tools.io.scip.load_scip>`
     - pyscipopt
   * - PIP
     - ``.pip``
     - :func:`load_scip <cpmpy.tools.io.scip.load_scip>`
     - pyscipopt
   * - XCSP3
     - ``.xml``
     - :func:`load_xcsp3 <cpmpy.tools.io.xcsp3.load_xcsp3>`
     - —
   * - JSPLib
     - (none)
     - :func:`load_jsplib <cpmpy.tools.io.jsplib.load_jsplib>`
     - —
   * - PSPLib (RCPSP)
     - ``.sm``
     - :func:`load_rcpsp <cpmpy.tools.io.rcpsp.load_rcpsp>`
     - —
   * - Nurse Rostering
     - ``.txt``
     - :func:`load_nurserostering <cpmpy.tools.io.nurserostering.load_nurserostering>`
     - —

Format-Specific Loaders
-----------------------

All format-specific loaders accept a file path *or* a raw content string.
This makes them usable both for on-disk files and for programmatically generated
or in-memory content.

.. code-block:: python

    # Load from file
    from cpmpy.tools.io.opb import load_opb
    model = load_opb("instance.opb")

    # Load from raw string
    content = "* #variable= 2 #constraint= 1\nx1 + x2 >= 1 ;"
    model = load_opb(content)

.. code-block:: python

    from cpmpy.tools.io.wcnf import load_wcnf
    model = load_wcnf("instance.wcnf")

.. code-block:: python

    from cpmpy.tools.io.dimacs import load_dimacs
    model = load_dimacs("instance.cnf")

.. code-block:: python

    import lzma
    from cpmpy.tools.io.xcsp3 import load_xcsp3
    model = load_xcsp3("instance.xml.lzma", open=lzma.open)

.. code-block:: python

    # MPS / LP / CIP / FZN / GMS / PIP  (require pyscipopt)
    from cpmpy.tools.io.scip import load_scip
    model = load_scip("instance.mps")
    model = load_scip("instance.lp")
    model = load_scip("instance.fzn")

.. code-block:: python

    from cpmpy.tools.io.jsplib import load_jsplib
    model = load_jsplib("instance")          # Job Shop Scheduling

.. code-block:: python

    from cpmpy.tools.io.rcpsp import load_rcpsp
    model = load_rcpsp("instance.sm")        # Resource-Constrained Project Scheduling

.. code-block:: python

    from cpmpy.tools.io.nurserostering import load_nurserostering
    model = load_nurserostering("instance.txt")

Compressed Files
----------------

All loaders accept a custom ``open`` callable for transparent decompression:

.. code-block:: python

    import lzma
    from cpmpy.tools.io.opb import load_opb

    model = load_opb("instance.opb.xz", open=lzma.open)

The same pattern applies to other loaders. For example, DIMACS CNF:

.. code-block:: python

    import lzma
    from cpmpy.tools.io.dimacs import load_dimacs

    model = load_dimacs("instance.cnf.xz", open=lambda p, mode="r": lzma.open(p, "rt"))

Datasets handle this automatically via ``dataset.open``. See
:doc:`datasets` and :doc:`/reading_and_writing` for details.

Listing Available Formats
--------------------------

.. code-block:: python

    from cpmpy.tools.io import read_formats
    print(read_formats())
    # ['mps', 'lp', 'cip', 'fzn', 'gms', 'pip', 'dimacs', 'opb', 'wcnf']

API Reference
-------------

.. automodule:: cpmpy.tools.io.reader
    :members:
    :undoc-members:
