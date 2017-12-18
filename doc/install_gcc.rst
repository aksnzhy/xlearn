Install GCC or Clang
---------------------------

If you have already installed your compiler before, you can skip this step.

* On Cygwin, run ``setup.exe`` and install ``gcc`` and ``binutils``.
* On Debian/Ubuntu Linux, type the command: ::

      sudo apt-get install gcc binutils 

  to install GCC (or Clang) by using :: 

      sudo apt-get install clang 

* On FreeBSD, type the following command to install Clang :: 

      sudo pkg_add -r clang 

* On Mac OS X, install ``XCode`` gets you Clang.


Install CMake
---------------------------

If you have already installed CMake before, you can skip this step.

To install CMake from binary packages:

* On Cygwin, run ``setup.exe`` and install cmake.
* On Debian/Ubuntu Linux, type the command to install cmake: ::

      sudo apt-get install cmake

* On FreeBSD, type the command: ::
   
      sudo pkg_add -r cmake

On Mac OS X, if you have ``homebrew``, you can use the command :: 

     brew install cmake

or if you have ``MacPorts``, run :: 

     sudo port install cmake

You won't want to have both Homebrew and MacPorts installed.

 .. toctree::
   :hidden: