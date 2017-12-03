#!/usr/bin/sh

python setup.py sdist

# make sure you know what you gonna do, and uncomment the following line
# twine register dist/*.tar.gz
# twine upload dist/*.tar.gz
