#!/bin/bash

mkdir build
cd build
cmake ..
make 

# install python package

cd python-package
if command -v python; then
    python setup.py install
fi

