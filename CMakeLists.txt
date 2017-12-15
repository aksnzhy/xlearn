# Copyright (c) 2016 by contributors.
# Authour: Chao Ma (mctt90@gmail.com)
#
# This is the root CMakeLists.txt file of the xLearn R package.
#
project ("xLearn-R")

cmake_minimum_required(VERSION 3.0)

#-------------------------------------------------------------------------------
# Take almost all warnings;
# Do not generate debug symbols;
# Optimization level 3;
# Using c++11;
# Optimization on architecture
#-------------------------------------------------------------------------------
add_definitions("-Wall -Wno-sign-compare -O3 -std=c++11 -march=native 
-Wno-strict-aliasing -Wno-comment")

#-------------------------------------------------------------------------------
# Declare where our project will be installed.
#-------------------------------------------------------------------------------
set(CMAKE_INSTALL_PREFIX "xLearn-R")

#-------------------------------------------------------------------------------
# Set include paths and library paths.
#-------------------------------------------------------------------------------
include_directories(
  "${PROJECT_SOURCE_DIR}"
  "${PROJECT_BINARY_DIR}"
  "${PROJECT_SOURCE_DIR}/gtest/include"
  "/Library/Frameworks/R.framework/Resources/include"
)

link_directories(
  "${PROJECT_BINARY_DIR}/src/base"
  "${PROJECT_BINARY_DIR}/src/data"
  "${PROJECT_BINARY_DIR}/src/reader"
  "${PROJECT_BINARY_DIR}/src/score"
  "${PROJECT_BINARY_DIR}/src/loss"
  "${PROJECT_BINARY_DIR}/src/solver"
  "${PROJECT_BINARY_DIR}/src/c_api"
)

#-------------------------------------------------------------------------------
# Build
#-------------------------------------------------------------------------------
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

# Build static library
set(STA_DEPS c_api solver reader loss score data base)
add_library(xlearn_R STATIC ./src/xlearn_R.cc)
target_link_libraries(xlearn_R ${STA_DEPS})

# Build shared library
add_library(xlearn_R_shared SHARED ./src/xlearn_R.cc ../c_api/c_api.cc 
../c_api/c_api_error.cc 
../base/logging.cc ../base/stringprintf.cc ../base/split_string.cc 
../base/levenshtein_distance.cc ../base/timer.cc 
../data/model_parameters.cc 
../loss/loss.cc ../loss/squared_loss.cc ../loss/cross_entropy_loss.cc 
../loss/metric.cc 
../reader/parser.cc ../reader/file_splitor.cc ../reader/reader.cc 
../score/score_function.cc ../score/linear_score.cc ../score/fm_score.cc 
../score/ffm_score.cc 
../solver/checker.cc ../solver/trainer.cc 
../solver/inference.cc ../solver/solver.cc)

# Set properties
set_target_properties(xlearn_R_shared PROPERTIES OUTPUT_NAME "xlearn_R")
set_target_properties(xlearn_R_shared PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(xlearn_R PROPERTIES CLEAN_DIRECT_OUTPUT 1)