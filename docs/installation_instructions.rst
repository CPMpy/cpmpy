Installation instructions
=========================

CPMpy requires Python ``3.6`` or newer. The package is available on `PyPI <https://pypi.org/>`_.

The easiest way is to install using the 'pip3' command line package manager. In a terminal, run:

.. code-block:: bash

    $ pip3 install cpmpy

This will automatically also install the default 'ortools' solver.

If the previous command fails to execute, it may be due to the permission to install the package globally Python packages. If so, you can install it for your user only with:

.. code-block:: bash

    $ pip3 install cpmpy --user

CPMpy has regular small releases with updates and improvements, so it is a good habbit to regularly update, as follows:

.. code-block:: bash

    $ pip3 install -U cpmpy

Installing on M1 Apple silicon
------------------------------
Google does not provide a binary distribution for the or-tools package to use on Apple Silicon yet, and you would get:

.. code-block:: bash

    ERROR: Could not find a version that satisfies the requirement ortools>=5.0 (from cpmpy) (from versions: none)
    ERROR: No matching distribution found for ortools>=5.0

Follow our `M1 installation instructions <installation_M1.html>`_ to build OR-tools from source instead.


Installing from a git repository
--------------------------------
If you want the very latest, or perhaps from an in-development branch, you can install directly from github as follows:

.. code-block:: bash

    $ pip3 install git+https://github.com/cpmpy/cpmpy@master

(change 'master' to any other branch or commit hash)
