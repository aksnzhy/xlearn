# Set output library.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/distributed)

# Build static library
set(STA_DEPS base)
add_library(distributed STATIC parameter_server.cc)
target_link_libraries(distributed ${STA_DEPS})

# Build unittests.
set(LIBS distributed data base gtest)

add_executable(parameter_server_test parameter_server_test.cc)
target_link_libraries(parameter_server_test gtest_main ${LIBS})

# Install library and header files
install(TARGETS distributed DESTINATION lib/distributed)
FILE(GLOB HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADER_FILES} DESTINATION include/distributed)