Installation instructions
=========================

CPMpy requires Python ``3.6`` or newer. The package is available on `PyPI <https://pypi.org/>`_.

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

    $ pip install git+https://github.com/cpmpy/cpmpy@master

(change 'master' to any other branch or commit hash)


Installing a local copy
-----------------------
If you are developing CPMpy locally, you can run scripts from in the repository folder, and it will use the cpmpy/ folder as package instead of any installed one.

However, if you want to test some local changes to CPMpy that can only be tested by installing CPMpy, you can do that as follows from the repository folder:

.. code-block:: bash

   $ pip install .

