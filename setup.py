from setuptools import find_packages, setup

with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

setup(
    name='cpmpy',
    version='0.5.3',
    author='Tias Guns',
    author_email="tias.guns@kuleuven.be",
    license='Apache 2.0',
    description='A numpy-based library for modeling constraint programming problems',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tias/cppy",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'ortools>=5.0',
        'numpy>=1.5',
    ],
    #extra dependency, only needed if minizinc is to be used.
    extras_require={
        "FULL":  ["minizinc"],
    },
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)
