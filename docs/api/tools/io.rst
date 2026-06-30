IO (:mod:`cpmpy.tools.io`)
==========================

.. py:module:: cpmpy.tools.io

.. automodule:: cpmpy.tools.io
    :members:
    :undoc-members:
    :inherited-members:


Available formats
-----------------

.. list-table::
   :header-rows: 1

   * - Format
     - Read
     - Write
   * - ``cip``
     - ✓
     - ✓
   * - ``cnf``
     - ✓
     - ✓
   * - ``dimacs``
     - ✓
     - ✓
   * - ``fzn``
     - ✓
     - ✓
   * - ``gms``
     - ✓
     - ✓
   * - ``lp``
     - ✓
     - ✓
   * - ``mps``
     - ✓
     - ✓
   * - ``opb``
     - ✓
     - ✓
   * - ``pip``
     - ✓
     - ✓
   * - ``wcnf``
     - ✓
     - ✓
   * - ``xcsp3``
     - ✓
     -
   * - ``jsplib``
     - ✓
     -
   * - ``nurserostering``
     - ✓
     -
   * - ``rcpsp``
     - ✓
     -


Format-specific loaders and writers:

.. toctree::
    :maxdepth: 1

    io/opb
    io/scip
    io/wcnf
    io/dimacs
    io/xcsp3
    io/jsplib
    io/rcpsp
    io/nurserostering