# Set output library.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/R-package)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

# Build static library
set(STA_DEPS c_api solver reader loss score data base)
add_library(xlearn_R STATIC src/init.cc src/xlearn_R.cc)
target_link_libraries(xlearn_R ${STA_DEPS})

# Build shared library
add_library(xlearn_R_shared SHARED ../src/c_api/c_api.cc ../src/c_api/c_api_error.cc 
../src/base/logging.cc ../src/base/stringprintf.cc ../src/base/split_string.cc 
../src/base/levenshtein_distance.cc ../src/base/timer.cc 
../src/data/model_parameters.cc 
../src/loss/loss.cc ../src/loss/squared_loss.cc ../src/loss/cross_entropy_loss.cc 
../src/loss/metric.cc 
../src/reader/parser.cc ../src/reader/file_splitor.cc ../src/reader/reader.cc 
../src/score/score_function.cc ../src/score/linear_score.cc ../src/score/fm_score.cc 
../src/score/ffm_score.cc 
../src/solver/checker.cc ../src/solver/trainer.cc 
../src/solver/inference.cc ../src/solver/solver.cc)

# Set properties
set_target_properties(xlearn_R_shared PROPERTIES OUTPUT_NAME "xlearn_R")
set_target_properties(xlearn_R_shared PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(xlearn_R PROPERTIES CLEAN_DIRECT_OUTPUT 1)