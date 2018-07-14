#!/bin/bash

mkdir build
cd build
cmake ..
make 

# install python package

cd python-package
if command -v python2; then
    python2 setup.py install
fi

if command -v python3; then
    python3 setup.py install
fi
