#!/bin/bash
# To be run from within manylinux2014 docker container.
# Run the command below from the root of the source folder:
# sudo docker run -ti -v $(pwd):/io quay.io/pypa/manylinux2014_x86_64 ./dev/build_wheels_manylinux2014.sh
# Or, to use singularity instead of docker (e.g. on HPC, or where root not available):
# singularity run -B $(pwd):/io -W /tmp -c docker://quay.io/pypa/manylinux2014_x86_64 /io/dev/build_wheels_manylinux2014.sh

set -e
cd /io || exit
PLAT=manylinux2014_x86_64

# Numpy provides manylinux2010 wheels only for Python up to 3.9.

/opt/python/cp310-cp310/bin/python -m build .
/opt/python/cp311-cp311/bin/python -m build .
/opt/python/cp312-cp312/bin/python -m build .

auditwheel repair dist/cherab-solps-*-cp310-cp310-linux_x86_64.whl --plat $PLAT
auditwheel repair dist/cherab-solps-*-cp311-cp311-linux_x86_64.whl --plat $PLAT
auditwheel repair dist/cherab-solps-*-cp312-cp312-linux_x86_64.whl --plat $PLAT
