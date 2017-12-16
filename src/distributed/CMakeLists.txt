set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/loss)

add_library(distributed STATIC dist_score_function.cc dist_linear_score.cc)

add_library(distributed_shared SHARED dist_score_function.cc dist_linear_score.cc)
