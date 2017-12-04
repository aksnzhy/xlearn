#!/usr/bin/sh

mkdir build
cd build
cmake ..
make 

# install python package

cd python-package
if command -v python2; then
    sudo python2 setup.py install
fi

if command -v python3; then
    sudo python3 setup.py install
fi


