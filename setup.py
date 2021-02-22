from setuptools import find_packages, setup

with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

setup(
    name='CP_Test_PY',
    version='0.0.1',
    author='VUB-Data-Lab',
    author_email="tias.guns@kuleuven.be",
    description='A numpy-based light-weight Python library for conveniently modeling constraint problems in Python',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tias/cppy",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'minizinc==0.4.2',
        'numpy==1.20.1',
        'ortools==8.1.8487',
        #Sub-dependencies
        'absl-py==0.11.0',
        'protobuf==3.15.1',
        'six==1.15.0'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)
