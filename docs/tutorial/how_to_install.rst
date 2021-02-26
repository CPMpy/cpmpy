Installation
============

CPMpy requires Python ``3.6`` or newer and is also installed with OR-Tools as default solver. Since the package is available on `PYPI <https://pypi.org/>`_, the best way is to open up a new command line window and run 

    $ pip install cpmpy

If the previous command fails to execute, it may be due to the permission to install the package globally Python packages. 

    $ pip install -U cpmpy

If Or-Tools is not installed yet or is not yet upgraded to the latest version, run the following command

    $ python -m pip install --upgrade --user ortools

Minizinc Installation and Configuration (optional)
--------------------------------------------------

A complete installation guide for the installation of Minizinc is available at `Minizinc Install Guide <https://www.minizinc.org/doc-2.5.3/en/installation.html#installation>`_

For **linux** users, if __snap__ is installed, Minizinc installation amounts to running

    $ snap install minizinc --classic

Otherwhise, the installation with the bundled Minizinc is straightforward (see https://www.minizinc.org/doc-2.5.3/en/installation.html#archive). The installation step boil down to:

1. Go to `Minzinc Download <https://www.minizinc.org/software.html>` and download the latest version of the binaries.
2. After downloading, *uncompress* the archive, for example in your home directory or any other location where you want to install it:

    tar xf MiniZincIDE-__replace_by_version__-bundle-linux-x86_64.tgz

3. Add the following lines to your `.bashrc` file (or `.zshrc` when using zsh):

    export PATH=MiniZincIDE-__replace_by_version__-bundle-linux-x86_64/bin:$PATH
    export LD_LIBRARY_PATH=MiniZincIDE-__replace_by_version__-bundle-linux-x86_64/lib:$LD_LIBRARY_PATH
    export QT_PLUGIN_PATH=MiniZincIDE-__replace_by_version__-bundle-linux-x86_64/plugins:$QT_PLUGIN_PATH
