Installation instructions
=========================

CPMpy requires Python ``3.6`` or newer. The package is available on `PYPI <https://pypi.org/>`_.

The easiest way is to install using the 'pip' command line package manager. In a terminal, run:

.. code-block:: bash

    $ pip install cpmpy

This will automatically also install the default 'ortools' solver.

If the previous command fails to execute, it may be due to the permission to install the package globally Python packages. If so, you can install it for your user only with:

.. code-block:: bash

    $ pip install cpmpy --user

CPMpy has regular small releases with updates and improvements, so it is a good habbit to regularly update, as follows:

.. code-block:: bash

    $ pip install -U cpmpy

Installing from a git repository
--------------------------------
If you want the very latest, or perhaps from an in-development branch, you can install directly from github as follows:

.. code-block:: bash

    $ pip3 install git+https://github.com/cpmpy/cpmpy@master

(change 'master' to any other branch or commit hash)
