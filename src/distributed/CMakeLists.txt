set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/loss)

add_library(distributed STATIC linear_score.cc)

add_library(distributed_shared SHARED linear_score.cc)
