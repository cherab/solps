#!/bin/sh

# Run in the PyPA manylinux2010 docker container to produce wheels suitable for PyPI.
# Ensure that the path to the cherab-solps repository is mounted at /cherab-solps.
# Python 3.10 and later should use the manylinux2014 docker container instead.

# Set cmake.args=--fresh to ensure no cached configuration from previous builds is used.

/opt/python/cp37-cp37m/bin/python -m build . -Ccmake.args=--fresh
/opt/python/cp38-cp38/bin/python -m build . -Ccmake.args=--fresh
/opt/python/cp39-cp39/bin/python -m build . -Ccmake.args=--fresh

for wheel in dist/*.whl
do
    auditwheel repair "$wheel"
done

# Upload the manylinux wheels, along with the sdist in dist/, using twine.
