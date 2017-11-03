# Copyright (c) 2016 by contributors.
# Authour: Chao Ma (mctt90@gmail.com)
#
# This is the root CMakeLists.txt file of the xLearn project, which
# build all subdirectories (packages) in the order of
# inter-package dependence.
#
# you can build the xLearn using the following commands:
#
# $> cd xLearn
# $> mkdir build
# $> cd build
# $> cmake ..
# $> make
#
# Thus you check out the xLearn project and build it in a
# subdirectory ``build". If you want further to install the
# built project, you can type the command:
#
# $> make install
#
project ("xLearn")

cmake_minimum_required(VERSION 2.8)

#-------------------------------------------------------------------------------
# Take almost all warnings;
# Take warnings as errors;
# Do not generate debug symbols;
# Optimization level 3;
# Using c++11;
# Using AVX instructions for speedup;
#-------------------------------------------------------------------------------
add_definitions("-Wall -Wno-sign-compare -Werror -O3 -std=c++11 -march=native")

#-------------------------------------------------------------------------------
# Declare where our project will be installed.
#-------------------------------------------------------------------------------
set(CMAKE_INSTALL_PREFIX "xLearn")

#-------------------------------------------------------------------------------
# Ensure executables are statically linked with libraries.
#-------------------------------------------------------------------------------
if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(CMAKE_EXE_LINKER_FLAGS "-static -static-libgcc")
endif()

#-------------------------------------------------------------------------------
# Set include paths and library paths.
#-------------------------------------------------------------------------------
include_directories(
  "${PROJECT_SOURCE_DIR}"
  "${PROJECT_SOURCE_DIR}/gtest/include"
)

link_directories(
  "${PROJECT_BINARY_DIR}"
)

#-------------------------------------------------------------------------------
# Declare packages in xLearn project.
#-------------------------------------------------------------------------------
add_subdirectory(gtest)
#add_subdirectory(demo)
add_subdirectory(src/base)
add_subdirectory(src/data)
#add_subdirectory(src/reader)
add_subdirectory(src/score)
add_subdirectory(src/loss)
#add_subdirectory(src/solver)
