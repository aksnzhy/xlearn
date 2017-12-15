#!/bin/bash

cp setup_pip.py setup.py
python setup.py sdist

# for testpypi
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# make sure you know what you gonna do, and uncomment the following line
# twine upload dist/*.tar.gz
