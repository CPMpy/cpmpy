# Deployment information

The full **_enter_name_of_package_**  documentation is available at [cpmpy](https://cpmpy.readthedocs.io/en/latest/index.html).

The documentation is automatically generated using Sphinx and deployed using [readthedocs.io](readthedocs.io).

## Requirements

Assuming you have Python already, install Sphinx:

    $ pip install sphinx

## Compiling the docs

Inside the `docs/` folder:

1. Clean the build folder, with the following command:

        make clean

2. Build the documentation

        make html

Once the documentation is successfully compiled and generated, push it to github and **readthedocs** automatically deploys the new documentation to the website.

