# Set output library.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/distributed)

# Build static library
set(STA_DEPS base)
add_library(distributed STATIC parameter_server.cc)
target_link_libraries(distributed ${STA_DEPS})