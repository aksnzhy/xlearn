set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/loss)

set(STA_DEPS data base)
add_library(distributed STATIC dist_score_function.cc dist_linear_score.cc)
target_link_libraries(distributed ${STA_DEPS})

set(SHA_DEPS data_shared base_shared)
add_library(distributed_shared SHARED dist_score_function.cc dist_linear_score.cc)
target_link_libraries(distributed_shared ${SHA_DEPS})
