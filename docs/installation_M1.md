# Installation on Mac with Apple silicon

Google does not provide a binary distribution for the or-tools package to use on Apple Silicon yet. Therefore, installation of cpmpy using the pip package manager will run into a compatibilty error like shown below.

```console
$ pip3 install cpmpy
ERROR: Could not find a version that satisfies the requirement ortools>=5.0 (from cpmpy) (from versions: none)
ERROR: No matching distribution found for ortools>=5.0
```

The solution is to build OR-tools from source on your machine. Altough OR-tools can be build from source on M1, the SCIP library is not supported on Apple Silicon. Therefore, it requires OR-tools to be build without this library.
There is an ongoing issue thread [#2322](https://github.com/google/or-tools/issues/2332) containing this workaround.

We keep this guide updated to new releases of OR-tools and SCIP.

## Install guide

Install command line tools
```console
$ xcode-select --install
```

Install homebrew
```console
$ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
$ brew update
```

Install gmake, which is the homebrew-version of make.
```console
$ brew install make
```

Install the swig compiler
```
$ brew install swig
```

Download OR-tools
```console
$ git clone https://github.com/google/or-tools && cd or-tools
```

In some cases it might be required to manually install the protobuffer package:
```console
$ pip3 install protobuffer
```

Build from source
```console
gmake third_party USE_SCIP=OFF
gmake python USE_SCIP=OFF
gmake install_python
```
The provided tests to validate the python install can be run with the following command: `gmake test_python`. However, this will fail upon testing the SCIP solver interface as it is not installed.

This should allow you to install cpmpy using the pip package manager as normal.
```console
pip3 install cpmpy
```
