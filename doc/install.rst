Installation Guide
----------------------------------

For now, xLearn can support both Linux and Mac OS X. We will support it on Windows platform in the near 
future. This page gives instructions on how to build and install the xLearn using ``pip`` and how to build 
it from source code. No matter what way you choose, make sure that your OS has already installed ``GCC`` or ``Clang`` 
(with the support of ``C++ 11``) and ``CMake``. 

Install GCC or Clang
^^^^^^^^^^^^^^^^^^^^^^^^

*If you have already installed your C++ compiler before, you can skip this step.*

* On Cygwin, run ``setup.exe`` and install ``gcc`` and ``binutils``.
* On Debian/Ubuntu Linux, type the command: ::

      sudo apt-get install gcc binutils 

  to install GCC (or Clang) by using: :: 

      sudo apt-get install clang 

* On FreeBSD, type the following command to install Clang: :: 

      sudo pkg_add -r clang 

* On Mac OS X, install ``XCode`` gets you Clang.


Install CMake
^^^^^^^^^^^^^^^^^^^^^^^^

*If you have already installed CMake before, you can skip this step.*

* On Cygwin, run ``setup.exe`` and install cmake.
* On Debian/Ubuntu Linux, type the command to install cmake: ::

      sudo apt-get install cmake

* On FreeBSD, type the command: ::
   
      sudo pkg_add -r cmake

On Mac OS X, if you have ``homebrew``, you can use the command: :: 

     brew install cmake

or if you have ``MacPorts``, run: :: 

     sudo port install cmake

You won't want to have both Homebrew and MacPorts installed.

Install xLearn from Source Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Building xLearn from source code consists two steps:

First, you need to build the executable files (``xlearn_train`` and ``xlearn_predict``), as well as the 
shared library (``libxlearn_api.so`` for Linux or ``libxlearn_api.dylib`` for Mac OSX) from the C++ code. After that, users need to install the xLearn Python Package.

Build from Source Code
=======================

Users need to clone the code from github: ::

  git clone https://github.com/aksnzhy/xlearn.git

  cd xlearn
  mkdir build
  cd build
  cmake ../
  make

If the building is successful, users can find two executable files (``xlearn_train`` and ``xlearn_predict``) in the ``build`` path. Users can test the installation by using the following command: ::

  ./run_example.sh

Install Python Package
=======================

Then, you can install the Python package through ``install-python.sh``: ::

  cd python-package
  sudo ./install-python.sh

You can also test the Python package by using the following command: ::

  cd ../
  python test_python.py

One-Button Building
=======================

We have already write a script ``build.sh`` to do all the cumbersome work for users, and users can just use the folloing commands: ::

  git clone https://github.com/aksnzhy/xlearn.git

  cd xlearn
  sudo ./build.sh

You may be asked to input your password during installation.

Install xLearn from pip
^^^^^^^^^^^^^^^^^^^^^^^^

The easiest way to install xLearn Python package is to use ``pip``. The following command will 
download the xLearn source code from pip and install Python package locally. You must make sure that you have already installed C++11 and CMake in your local machine: ::

    sudo pip install xlearn

The installation process will take a while to complete. After that, you can type the following script in your python shell to check whether the xLearn has been installed successfully: ::

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

        xLearn   -- 0.43 Version --
  -------------------------------------------------------------------------


Install R Package
^^^^^^^^^^^^^^^^^^^^^^^^

The R package installation guide is coming soon.
