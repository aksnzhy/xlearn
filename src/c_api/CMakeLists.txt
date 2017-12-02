# Set output library.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/c_api)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

# Build static library
set(STA_DEPS solver reader loss score data base)
add_library(xlearn STATIC c_api.cc c_api_error.cc)
target_link_libraries(xlearn ${STA_DEPS})

# Build shared library
set(SHA_DEPS solver reader loss score data base)
add_library(xlearn_shared SHARED c_api.cc c_api_error.cc)
target_link_libraries(xlearn_shared ${SHA_DEPS})

# Set properties
set_target_properties(xlearn_shared PROPERTIES OUTPUT_NAME "xlearn")
set_target_properties(xlearn_shared PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(xlearn PROPERTIES CLEAN_DIRECT_OUTPUT 1)

# Build unittests.
set(LIBS xlearn solver reader loss score data base pthread gtest)

add_executable(c_api_test c_api_test.cc)
target_link_libraries(c_api_test gtest_main ${LIBS})

# Copy Python file
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/c_api_test.py" 
DESTINATION ${PROJECT_BINARY_DIR}/test/c_api)

# Install library and header files
install(TARGETS xlearn DESTINATION lib/c_api)
FILE(GLOB HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADER_FILES} DESTINATION include/c_api)