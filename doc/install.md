## Installation Guide

For now, the xLearn can support Linux and Mac osx. This page gives instructions on how to build and install the xLearn package from source code. It consists of two steps:

 1. First build the executable files (`xlearn_train` and `xlearn_predict`) and shared library (`libxlearn.so` for Linux and `libxlearn.dylib` for Mac osx) from the C++ codes.
 2. Then install the python package.

### Build executable file and shared libaray

Our goal is to build the shared libary:

 - On Linux the target library is `libxlearn.so`
 - On Mac osx the target library is `libxlearn.dylib`

Also, we will build the executable files, which can be used in commad line. The executable files include:

 - `xlearn_train` is for training task.
 - `xlearn_predict` is for prediction task.

xLearn doesn't rely on any thrid-party library and hence users can just clone the code and compile it by using `cmake`. To compile xLearn we need a c++ compiler supoorting `C++ 11` (e.g., g++4.8 or higher)

#### Install GCC or Clang

If you have already installed your compiler before, you can skip this step.

  * On Cygwin, run `setup.exe` and install `gcc` and `binutils`.
  * On Debian/Ubuntu Linux, type the command `sudo apt-get install gcc binutils` to install GCC, or `sudo apt-get install clang` to install Clang.
  * On FreeBSD, type the command `sudo pkg_add -r clang` to install Clang.  Note that since version 9.0, FreeBSD does not update GCC but relies completely on Clang.
  * On Mac OS X, install XCode gets you Clang.

#### Install CMake

If you have already installed CMake before, you can skip this step.

To install CMake from binary packages:

  * On Cygwin, run `setup.exe` and install `cmake`.
  * On Debian/Ubuntu Linux, type the command `sudo apt-get install cmake`.
  * On FreeBSD, type the command `sudo pkg_add -r cmake`.
  * On Mac OS X, if you have [http://mxcl.github.com/homebrew/ Howebew], you can use the command `brew install cmake`, or if you have [http://www.macports.org/ MacPorts], run `sudo port install cmake`.  You won't want to have both Homebrew and !MacPorts installed.

You can also download binary or source package of [CMake](http://www.cmake.org/cmake/resources/software.html) and install it manually.

#### Build xLearn

Now you can build xLearn. First clone the repositpory:

    git clone https://github.com/aksnzhy/xlearn.git

and then build using the following commands:

    cd xlearn; mkdir build; cd build
    cmake ..
    make -j4

### Build python package

    