Installation Guide for Windows
----------------------------------

For now, xLearn can support both Linux, Mac OS X and Windows. This page gives instructions on how to build and install the xLearn
from source code on Windows. Before starting,  make sure that your Windows has already installed  ``Visual Studio 2017`` and ``CMake``. 

Install Visual Studio 2017
^^^^^^^^^^^^^^^^^^^^^^^^

*If you have already installed your C++ compiler before, you can skip this step.*

Download Visual Studio ``vs_xxxx_xxxx.exe`` from https://visualstudio.microsoft.com/downloads/, then you can follow the VS2017 install guide
https://docs.microsoft.com/en-us/visualstudio/install/install-visual-studio?view=vs-2017. Users should make sure that choose the c++
development tools when install VS2017.
 
Install CMake
^^^^^^^^^^^^^^^^^^^^^^^^

*If you have already installed CMake before, you can skip this step.*

Download latest(at least v3.10) package for windows from https://cmake.org/download/ and then install it. whether you choose ``.msi`` or ``.zip`` package, 
you should make sure that cmake is added to your system path.

Install xLearn from Source Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Building xLearn from source code consists two steps:

First, you need to build the executable files (``xlearn_train.exe`` and ``xlearn_predict.exe``), as well as the 
shared library (``xlearn_api.dll`` for Windows) from the C++ code. After that, users need to install the xLearn Python Package.

Build from Source Code
=======================
First, users should enter DOS as Administrator. 
Then, users need to clone the code from github: ::

  git clone https://github.com/aksnzhy/xlearn.git

  cd xlearn
  mkdir build
  cd build
  cmake -G "Visual Studio 15 Win64" ../
  "C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
  MSBuild xLearn.sln /p:Configuration=Release
  
**Note:** You should replace this path ``"C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build\vcvarsall.bat"``
to yourself installation path of VS2017.

Suppose you install the VS Community version, the path should be ``"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat"``
if you install it in default path.

If the building is successful, users can find two executable files (``xlearn_train.exe`` and ``xlearn_predict.exe``) in the ``build\Release`` path. 
Users can test the installation by using the following command: ::

  run_example.bat

Install Python Package
=======================

Then, you can install the Python package through ``install-python.sh``: ::

  cd python-package
  python setup.py install 

You can also test the Python package by using the following command: ::

  cd ../
  python test_python.py

One-Button Building
=======================

We have already write a script ``build.bat`` to do all the cumbersome work for users, and users can just use the folloing commands: ::

  git clone https://github.com/aksnzhy/xlearn.git

  cd xlearn
  build.bat

You should make sure that you enter DOS as Administrator.

Install xLearn from pip
^^^^^^^^^^^^^^^^^^^^^^^^

We will update Python package for Windows soon later.

The installation process will take a while to complete. 
After that, you can type the following script in your python shell to check whether the xLearn has been installed successfully: ::

  >>> import xlearn as xl
  >>> xl.hello()

You will see the following message if the installation is successful: ::

  -------------------------------------------------------------------------
           _
          | |
     __  _| |     ___  __ _ _ __ _ __
     \ \/ / |    / _ \/ _` | '__| '_ \
      >  <| |___|  __/ (_| | |  | | | |
     /_/\_\_____/\___|\__,_|_|  |_| |_|

        xLearn   -- 0.41 Version --
  -------------------------------------------------------------------------
