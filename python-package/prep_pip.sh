#!/bin/bash

cp setup_pip.py setup.py
python setup.py sdist

# make sure you know what you gonna do, and uncomment the following line
# twine upload dist/*.tar.gz

