# Set output library.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/data)

# Build static library
set(STA_DEPS base)
add_library(data STATIC model_parameters.cc)
target_link_libraries(data ${STA_DEPS})

# Build unittests.
set(LIBS data base gtest)

add_executable(data_structure_test data_structure_test.cc)
target_link_libraries(data_structure_test gtest_main ${LIBS})

add_executable(model_parameters_test model_parameters_test.cc)
target_link_libraries(model_parameters_test gtest_main ${LIBS})

# Install library and header files
install(TARGETS data DESTINATION lib/data)
FILE(GLOB HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADER_FILES} DESTINATION include/data)
