## Installation Guide

------

For now, the xLearn can support Linux and Mac osx. This page gives instructions on how to build and install the xLearn package from source code. It consists of two steps:

 1. First build the executable files (***xlearn_train*** and ***xlearn_predict***) and shared library (***libxlearn.so*** for Linux and ***libxlearn.dylib*** for Mac osx) from the C++ codes.
 2. Then install the python package.

### Build executable file and shared libaray

Our goal is to build the shared libary:

 - On Linux the target library is ***libxlearn.so***
 - On Mac osx the target library is ***libxlearn.dylib***

Also, we will build the executable files, which can be used in commad line. The executable files include:

 - ***xlearn_train*** is for training task
 - ***xlearn_predict*** is for prediction task

 
### Build python package

