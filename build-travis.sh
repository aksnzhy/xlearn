#! /bin/bash

mkdir build
cd build
cmake ..
make

# install python package

cd python-package
if command -v python2; then
    sudo python2 setup.py install
fi
