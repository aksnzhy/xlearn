#!/bin/bash
# This script is for installization of xlearn python package
# You may need to type your password here
sudo python setup.py install
# Reset the path of python library
basepath=$(cd `dirname $0`; pwd)
export PYTHONPATH=${basepath}/xlearn