#!/bin/sh

# Run in the PyPA manylinux1 docker container to produce wheels suitable for PyPI.
# Ensure that the path to the cherab-solps repository is mounted at /cherab-solps.

# Due to the lack of manylinux1 wheels available for many of the dependencies, we
# install build-time dependencies manually using the last versions with wheels,
# then run build without build-time isolation.

/opt/python/cp37-cp37m/bin/python -m venv /tmp/venv --clear
. /tmp/venv/bin/activate
pip install --prefer-binary "numpy==1.14.6" "raysect==0.7.1" build wheel "cherab==1.3.0" "scipy<1.6" "matplotlib<3.3" "cython==3.0a5"
python -m build -n .

/opt/python/cp38-cp38/bin/python -m venv /tmp/venv --clear
. /tmp/venv/bin/activate
pip install --prefer-binary "numpy==1.17.5" "raysect==0.7.1" build wheel "cherab==1.3.0" "scipy<1.8" "matplotlib<3.6" "cython==3.0a5"
python -m build -n .

/opt/python/cp39-cp39/bin/python -m venv /tmp/venv --clear
. /tmp/venv/bin/activate
pip install --prefer-binary "numpy==1.19.5" "raysect==0.7.1" build wheel "cherab==1.3.0" "scipy<1.8" "matplotlib<3.6" "cython==3.0a5"
python -m build -n .

for wheel in dist/*.whl
do
    auditwheel repair "$wheel"
done

# Upload the manylinux1 wheels, along with the sdist in dist/, using twine.
