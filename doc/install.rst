Installation Guide
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For now, xLearn can support Linux and Mac OS X. We will support it on Windows platform in 
the near future. This page gives instructions on how to build and install the xLearn 
package using pip and how to build it from source code. No matter what way you choose, make 
sure that your OS has already installed ``GCC`` (or ``Clang``) and ``CMake``, and your compiler 
need to support ``C++11``. If you have not installed them, please see `this page`__ on how to 
install GCC and CMake.

Install xLearn from pip
---------------------------

The easiest way to install xLearn Python package is to use ``pip``. The following command will 
download the xLearn source code from pip and install Python package locally.  ::

    sudo pip install xlearn

The installation process will take a while to complete. And then you can type the following 
script in your python shell to check whether the xLearn has been installed successfully:

>>> import xlearn as xl
>>> xl.hello()

You will see: ::

  -------------------------------------------------------------------------
           _
          | |
     __  _| |     ___  __ _ _ __ _ __
     \ \/ / |    / _ \/ _` | '__| '_ \
      >  <| |___|  __/ (_| | |  | | | |
     /_/\_\_____/\___|\__,_|_|  |_| |_|

        xLearn   -- 0.30 Version --
  -------------------------------------------------------------------------

If you want to build the latest code from `Github`__, or you want to use the xLearn command line 
instead of the Python API, you can see how to build xLearn from source code as follow. We highly
recommend that you can build xLearn from source code.

Install xLearn from Source Code
----------------------------------

Building xLearn from source code consists two steps.

First, you need to build the executable files (``xlearn_train`` and ``xlearn_predict``), as well as the 
shared library (``libxlearn_api.so`` for Linux and ``libxlearn_api.dylib`` for Mac OSX) from the C++ code.

Then, you can install the Python package through ``install-python.sh``.

Fortunately, we write a script ``build.sh`` to do all the cumbersome work for users.

For users, you just need to clone the code from github ::

  git clone https://github.com/aksnzhy/xlearn.git

and then build xLearn using the folloing commands: ::

  cd xlearn
  ./build.sh

You may be asked to input your password during installation.

Test Your Building
----------------------------------

Now you can test your installation by using the following command: ::

  cd build
  ./run_example.sh

You can also test the Python package by using the following command: ::

  cd python-package/test
  python test_python.py

Install R Package
----------------------------------

The R package installation guide is coming soon.

.. __: install_cmake.html
.. __: https://github.com/aksnzhy/xlearn

 .. toctree::
   :hidden:

   install_cmake.rst